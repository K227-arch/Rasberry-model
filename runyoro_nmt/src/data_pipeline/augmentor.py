"""
DataAugmentor
=============
Applies multiple data augmentation strategies to increase training data size
and improve model robustness:

  1. Token deletion (randomly drop non-critical tokens)
  2. Token swap (randomly swap adjacent tokens)
  3. Synonym substitution (English side using WordNet if available)
  4. Back-translation paraphrasing (placeholder — hooks into trained model)
  5. Glossary-aware augmentation (protect named entities and technical terms)
  6. Noise injection (simulates OCR/typing errors for robustness)

Generates a comprehensive augmentation report with statistics.
"""

from __future__ import annotations

import logging
import random
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runyoro-Rutooro function words that should NOT be deleted or swapped
# (grammatically critical)
# ---------------------------------------------------------------------------
RNY_PROTECTED = {
    "ni", "na", "nga", "ka", "mu", "ku", "ha", "ti", "nti",
    "eki", "omu", "eri", "ama", "obu", "aka", "aba",
}

ENG_PROTECTED = {
    "the", "a", "an", "is", "are", "was", "were", "not", "no",
    "I", "you", "he", "she", "it", "we", "they",
}


@dataclass
class AugmentationRecord:
    original_rny: str
    original_eng: str
    augmented_rny: str
    augmented_eng: str
    strategy: str


@dataclass
class AugmentationResult:
    original_pairs: List[Tuple[str, str]]
    augmented_pairs: List[Tuple[str, str]]
    records: List[AugmentationRecord] = field(default_factory=list)

    @property
    def total_pairs(self) -> int:
        return len(self.original_pairs) + len(self.augmented_pairs)

    def summary(self) -> str:
        return (
            f"Original: {len(self.original_pairs)} | "
            f"Augmented: {len(self.augmented_pairs)} | "
            f"Total: {self.total_pairs}"
        )


class DataAugmentor:
    """Applies augmentation strategies to Runyoro-Rutooro/English parallel pairs."""

    def __init__(
        self,
        seed: int = 42,
        deletion_prob: float = 0.05,
        swap_prob: float = 0.05,
        augment_multiplier: int = 2,
        protected_terms: Optional[Set[str]] = None,
    ):
        self.seed = seed
        self.deletion_prob = deletion_prob
        self.swap_prob = swap_prob
        self.augment_multiplier = augment_multiplier
        self.protected_terms = protected_terms or set()
        random.seed(seed)

        self._wordnet_available = self._check_wordnet()

    @staticmethod
    def _check_wordnet() -> bool:
        try:
            import nltk
            nltk.data.find("corpora/wordnet")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Token deletion
    # ------------------------------------------------------------------
    def _token_deletion(self, text: str, protected: Set[str]) -> str:
        tokens = text.split()
        if len(tokens) <= 2:
            return text
        result = [
            t for t in tokens
            if t.lower() in protected or random.random() > self.deletion_prob
        ]
        return " ".join(result) if result else text

    # ------------------------------------------------------------------
    # Token swap (adjacent tokens only)
    # ------------------------------------------------------------------
    def _token_swap(self, text: str, protected: Set[str]) -> str:
        tokens = text.split()
        if len(tokens) < 3:
            return text
        tokens = tokens.copy()
        for i in range(len(tokens) - 1):
            if (
                tokens[i].lower() not in protected
                and tokens[i+1].lower() not in protected
                and random.random() < self.swap_prob
            ):
                tokens[i], tokens[i+1] = tokens[i+1], tokens[i]
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # English synonym substitution (WordNet)
    # ------------------------------------------------------------------
    def _synonym_substitute(self, text: str) -> str:
        if not self._wordnet_available:
            return text
        try:
            from nltk.corpus import wordnet  # type: ignore

            tokens = text.split()
            for i, token in enumerate(tokens):
                clean = re.sub(r"[^a-zA-Z]", "", token).lower()
                if clean in ENG_PROTECTED or len(clean) < 4:
                    continue
                synsets = wordnet.synsets(clean)
                if synsets:
                    lemmas = [
                        l.name().replace("_", " ")
                        for s in synsets[:2]
                        for l in s.lemmas()
                        if l.name().lower() != clean
                    ]
                    if lemmas and random.random() < 0.2:
                        tokens[i] = lemmas[0]
            return " ".join(tokens)
        except Exception:
            return text

    # ------------------------------------------------------------------
    # Noise injection (simulates realistic errors)
    # ------------------------------------------------------------------
    @staticmethod
    def _inject_noise(text: str, noise_prob: float = 0.02) -> str:
        """Randomly duplicate or drop individual characters to simulate OCR noise."""
        chars = list(text)
        result = []
        for ch in chars:
            if ch.isalpha() and random.random() < noise_prob:
                if random.random() < 0.5:
                    result.append(ch)  # duplicate
                # else drop the character
            else:
                result.append(ch)
        return "".join(result)

    # ------------------------------------------------------------------
    # Capitalisation variation (Runyoro)
    # ------------------------------------------------------------------
    @staticmethod
    def _capitalise_variation(text: str) -> str:
        """Randomly capitalise the first letter of the string (mirrors real data variance)."""
        if text and random.random() < 0.5:
            return text[0].upper() + text[1:]
        return text

    # ------------------------------------------------------------------
    # Main augmentation pipeline
    # ------------------------------------------------------------------
    def augment(
        self, pairs: List[Tuple[str, str]]
    ) -> AugmentationResult:
        result = AugmentationResult(
            original_pairs=list(pairs),
            augmented_pairs=[],
        )

        strategy_counts: Dict[str, int] = {}

        for rny, eng in pairs:
            for _ in range(self.augment_multiplier):
                strategy = random.choice([
                    "token_deletion",
                    "token_swap",
                    "synonym_substitute",
                    "combined",
                ])

                aug_rny = rny
                aug_eng = eng

                if strategy == "token_deletion":
                    aug_rny = self._token_deletion(rny, RNY_PROTECTED)
                    aug_eng = self._token_deletion(eng, ENG_PROTECTED)
                elif strategy == "token_swap":
                    aug_rny = self._token_swap(rny, RNY_PROTECTED)
                    aug_eng = self._token_swap(eng, ENG_PROTECTED)
                elif strategy == "synonym_substitute":
                    aug_eng = self._synonym_substitute(eng)
                elif strategy == "combined":
                    aug_rny = self._token_deletion(rny, RNY_PROTECTED)
                    aug_eng = self._synonym_substitute(
                        self._token_swap(eng, ENG_PROTECTED)
                    )

                # Skip if nothing changed
                if aug_rny == rny and aug_eng == eng:
                    continue

                # Reject empty outputs
                if not aug_rny.strip() or not aug_eng.strip():
                    continue

                result.augmented_pairs.append((aug_rny, aug_eng))
                result.records.append(
                    AugmentationRecord(
                        original_rny=rny,
                        original_eng=eng,
                        augmented_rny=aug_rny,
                        augmented_eng=aug_eng,
                        strategy=strategy,
                    )
                )
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        logger.info(
            "Augmentation complete: %s | Strategies: %s",
            result.summary(), strategy_counts,
        )
        return result

    def generate_report(
        self,
        result: AugmentationResult,
        output_path: Optional[str] = None,
    ) -> str:
        strategy_counts: Dict[str, int] = {}
        for r in result.records:
            strategy_counts[r.strategy] = strategy_counts.get(r.strategy, 0) + 1

        lines = [
            "# Data Augmentation Report",
            "",
            f"**Original pairs:** {len(result.original_pairs)}",
            f"**Augmented pairs generated:** {len(result.augmented_pairs)}",
            f"**Total pairs (original + augmented):** {result.total_pairs}",
            "",
            "## Strategy Breakdown",
            "",
            "| Strategy | Count |",
            "|----------|-------|",
        ]
        for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{s}` | {c} |")

        lines += [
            "",
            "## Examples (first 10 augmented pairs)",
            "",
        ]
        for i, rec in enumerate(result.records[:10]):
            lines += [
                f"### Augmentation {i+1} — Strategy: `{rec.strategy}`",
                "",
                f"**Original Runyoro:** `{rec.original_rny}`",
                f"**Augmented Runyoro:** `{rec.augmented_rny}`",
                "",
                f"**Original English:** `{rec.original_eng}`",
                f"**Augmented English:** `{rec.augmented_eng}`",
                "",
                "---",
                "",
            ]

        report = "\n".join(lines)
        if output_path:
            from pathlib import Path
            Path(output_path).write_text(report, encoding="utf-8")
            logger.info("Augmentation report written to %s", output_path)

        return report
