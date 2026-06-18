"""
NMTEvaluator
============
Computes a full suite of automatic translation quality metrics:

  - BLEU (sacreBLEU)
  - chrF++ (character F-score with word order)
  - TER (Translation Edit Rate)
  - BERTScore
  - COMET
  - Back-translation consistency

Also runs a back-translation roundtrip check and generates a
comprehensive evaluation report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)


class NMTEvaluator:
    """
    Evaluates translation quality using multiple automatic metrics.
    """

    def __init__(
        self,
        model_path: str,
        src_lang: str = "nyk_Latn",
        tgt_lang: str = "eng_Latn",
        beam_size: int = 4,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.beam_size = beam_size
        self.device = device

        self._model = None
        self._tokenizer = None
        self._metrics_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
            import torch

            logger.info("Loading model for evaluation: %s", self.model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path
            ).to(self.device)
            self._model.eval()
        except Exception as e:
            logger.error("Model load failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    def translate_batch(
        self, sources: List[str], src_lang: str, tgt_lang: str
    ) -> List[str]:
        self._load_model()
        import torch

        self._tokenizer.src_lang = src_lang
        forced_bos_id = self._tokenizer.lang_code_to_id[tgt_lang]

        results = []
        batch_size = 8
        for i in range(0, len(sources), batch_size):
            batch = sources[i:i+batch_size]
            enc = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)

            with torch.no_grad():
                generated = self._model.generate(
                    **enc,
                    forced_bos_token_id=forced_bos_id,
                    num_beams=self.beam_size,
                    max_length=256,
                    length_penalty=1.0,
                )
            decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
            results.extend(decoded)

        return results

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _compute_bleu(
        self, predictions: List[str], references: List[str]
    ) -> Dict:
        try:
            import evaluate  # type: ignore
            metric = evaluate.load("sacrebleu")
            result = metric.compute(
                predictions=predictions,
                references=[[r] for r in references],
            )
            return {"bleu": round(result["score"], 2), "bp": round(result["bp"], 4)}
        except Exception as e:
            logger.warning("BLEU failed: %s", e)
            return {"bleu": -1}

    def _compute_chrf(
        self, predictions: List[str], references: List[str]
    ) -> Dict:
        try:
            import evaluate  # type: ignore
            metric = evaluate.load("chrf")
            result = metric.compute(
                predictions=predictions,
                references=[[r] for r in references],
                word_order=2,
            )
            return {"chrf": round(result["score"], 2)}
        except Exception as e:
            logger.warning("chrF++ failed: %s", e)
            return {"chrf": -1}

    def _compute_ter(
        self, predictions: List[str], references: List[str]
    ) -> Dict:
        try:
            import evaluate  # type: ignore
            metric = evaluate.load("ter")
            result = metric.compute(
                predictions=predictions,
                references=[[r] for r in references],
                normalized=True,
            )
            return {"ter": round(result["score"], 2)}
        except Exception as e:
            logger.warning("TER failed: %s", e)
            return {"ter": -1}

    def _compute_bertscore(
        self, predictions: List[str], references: List[str]
    ) -> Dict:
        try:
            import evaluate  # type: ignore
            metric = evaluate.load("bertscore")
            result = metric.compute(
                predictions=predictions,
                references=references,
                lang="en",
            )
            avg_f1 = sum(result["f1"]) / len(result["f1"])
            return {"bertscore_f1": round(avg_f1, 4)}
        except Exception as e:
            logger.warning("BERTScore failed: %s", e)
            return {"bertscore_f1": -1}

    def _compute_comet(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str],
    ) -> Dict:
        try:
            import evaluate  # type: ignore
            metric = evaluate.load("comet")
            result = metric.compute(
                predictions=predictions,
                references=references,
                sources=sources,
            )
            return {"comet": round(result["mean_score"], 4)}
        except Exception as e:
            logger.warning("COMET failed: %s", e)
            return {"comet": -1}

    # ------------------------------------------------------------------
    # Back-translation consistency
    # ------------------------------------------------------------------
    def back_translation_score(
        self, pairs: List[Tuple[str, str]], sample_size: int = 50
    ) -> Dict:
        """
        Translate src→tgt→src and measure roundtrip similarity.
        High score = model is internally consistent.
        """
        import random
        sample = random.sample(pairs, min(sample_size, len(pairs)))
        sources = [r for r, e in sample]
        references = sources  # we want to recover the original

        # Forward: rny → en
        forward = self.translate_batch(sources, self.src_lang, self.tgt_lang)
        # Backward: en → rny
        roundtrip = self.translate_batch(forward, self.tgt_lang, self.src_lang)

        # Simple char overlap score
        scores = []
        for orig, rt in zip(sources, roundtrip):
            orig_chars = set(orig.lower())
            rt_chars = set(rt.lower())
            if not orig_chars:
                continue
            overlap = len(orig_chars & rt_chars) / len(orig_chars)
            scores.append(overlap)

        avg = sum(scores) / len(scores) if scores else 0.0
        return {
            "back_translation_roundtrip_score": round(avg, 4),
            "sample_size": len(sample),
        }

    # ------------------------------------------------------------------
    # Full evaluation run
    # ------------------------------------------------------------------
    def evaluate(
        self,
        test_pairs: List[Tuple[str, str]],
        output_dir: Optional[str] = None,
    ) -> Dict:
        logger.info("Running full evaluation on %d test pairs...", len(test_pairs))

        sources = [r for r, e in test_pairs]
        references = [e for r, e in test_pairs]

        predictions = self.translate_batch(sources, self.src_lang, self.tgt_lang)

        results: Dict = {}
        results.update(self._compute_bleu(predictions, references))
        results.update(self._compute_chrf(predictions, references))
        results.update(self._compute_ter(predictions, references))
        results.update(self._compute_bertscore(predictions, references))
        results.update(self._compute_comet(sources, predictions, references))
        results.update(self.back_translation_score(test_pairs))

        results["n_test_pairs"] = len(test_pairs)

        logger.info("Evaluation results: %s", results)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "eval_results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
            # Save prediction examples
            examples = [
                {"source": s, "prediction": p, "reference": r}
                for s, p, r in zip(sources[:50], predictions[:50], references[:50])
            ]
            (out / "eval_predictions_sample.json").write_text(
                json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            logger.info("Evaluation results saved to %s", output_dir)

        return results
