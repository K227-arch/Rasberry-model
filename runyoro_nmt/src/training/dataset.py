"""
ParallelDataset & DataCollator
==============================
PyTorch Dataset and HuggingFace-compatible DataCollator for the
Runyoro-Rutooro / English NMT training pipeline.

Supports:
  - Bidirectional training (rny→en AND en→rny in same epoch)
  - Domain-weighted sampling
  - NLLB-200 language token injection
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Optional, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase  # type: ignore

logger = logging.getLogger(__name__)

# NLLB-200 language token IDs
# Runyoro-Rutooro is closest to "nyk_Latn" (Nyankore) in NLLB
NLLB_RNY = "nyk_Latn"
NLLB_ENG = "eng_Latn"


class ParallelDataset(Dataset):
    """
    Dataset for bidirectional Runyoro-Rutooro / English NMT.

    Each item yields a (source, target) sentence pair.
    When `bidirectional=True`, the dataset is doubled:
      - first half:  Runyoro → English
      - second half: English → Runyoro
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        src_lang: str = NLLB_RNY,
        tgt_lang: str = NLLB_ENG,
        max_source_length: int = 256,
        max_target_length: int = 256,
        bidirectional: bool = True,
        domain_weights: Optional[Dict[str, float]] = None,
    ):
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.bidirectional = bidirectional

        # Build forward (rny→en) samples
        self._forward: List[Tuple[str, str, str, str]] = [
            (rny, eng, src_lang, tgt_lang) for rny, eng in pairs
        ]

        if bidirectional:
            # Also add reverse (en→rny) samples
            reverse = [
                (eng, rny, tgt_lang, src_lang) for rny, eng in pairs
            ]
            self._all_samples = self._forward + reverse
        else:
            self._all_samples = self._forward

        logger.info(
            "Dataset created: %d total samples (%s bidirectional)",
            len(self._all_samples),
            "with" if bidirectional else "without",
        )

    def __len__(self) -> int:
        return len(self._all_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src_text, tgt_text, src_lang, tgt_lang = self._all_samples[idx]

        # Tokenise source
        self.tokenizer.src_lang = src_lang
        source_enc = self.tokenizer(
            src_text,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        # Tokenise target (forced BOS = target language token)
        with self.tokenizer.as_target_tokenizer():  # type: ignore[attr-defined]
            target_enc = self.tokenizer(
                tgt_text,
                max_length=self.max_target_length,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )

        labels = target_enc["input_ids"].squeeze()
        # Replace padding token id with -100 (ignored in loss)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": labels,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
        }


class DataCollatorForNMT:
    """
    Dynamic padding collator for NMT.
    Pads sequences within a batch to the maximum length in that batch
    rather than a global max (more memory efficient).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        label_pad_token_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Dynamic padding
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.label_pad_token_id
        )

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_masks_padded,
            "labels": labels_padded,
        }
