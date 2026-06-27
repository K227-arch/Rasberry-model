"""
ParallelDataset & DataCollator
==============================
PyTorch Dataset and HuggingFace-compatible DataCollator for the
Runyoro-Rutooro / English NMT training pipeline.

NO LANGUAGE CODES — trains as plain seq2seq without NLLB language tokens.
The model learns the mapping directly from Runyoro text patterns to English
and vice versa, without relying on language codes that don't exist in the
NLLB vocabulary for Runyoro.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Optional, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase  # type: ignore

logger = logging.getLogger(__name__)

# No language codes - train as plain seq2seq
NLLB_RNY = None  # Not used
NLLB_ENG = None  # Not used


class ParallelDataset(Dataset):
    """
    Dataset for bidirectional Runyoro-Rutooro / English NMT.
    No language codes — plain text-to-text.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        src_lang: str = None,
        tgt_lang: str = None,
        max_source_length: int = 256,
        max_target_length: int = 256,
        bidirectional: bool = True,
        domain_weights: Optional[Dict[str, float]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.bidirectional = bidirectional

        # Build samples
        self._forward: List[Tuple[str, str]] = [(rny, eng) for rny, eng in pairs]

        if bidirectional:
            reverse = [(eng, rny) for rny, eng in pairs]
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
        src_text, tgt_text = self._all_samples[idx]

        # Tokenise source — no language code
        source_enc = self.tokenizer(
            src_text,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        # Tokenise target — no language code
        target_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source_enc["input_ids"].squeeze(),
            "attention_mask": source_enc["attention_mask"].squeeze(),
            "labels": labels,
        }


class DataCollatorForNMT:
    """
    Dynamic padding collator for NMT.
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
