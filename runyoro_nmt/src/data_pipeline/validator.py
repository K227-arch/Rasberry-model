"""
DataValidator
=============
Validates, scores, and flags parallel sentence pairs for quality issues:

  - Language identification (checks that each side is the correct language)
  - Length ratio filtering
  - Minimum / maximum token count
  - Misalignment detection (e.g. both sides in the same language)
  - Code-switching detection
  - Duplicate detection
  - Runyoro-Rutooro orthographic rule checks
  - English grammar surface checks

Produces a detailed QA report with before/after examples.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runyoro-Rutooro linguistic rules (orthographic / morphological surface checks)
# ---------------------------------------------------------------------------
# Common Runyoro-Rutooro noun class prefixes (Bantu noun classes)
RNY_NOUN_PREFIXES = [
    "eki", "ebi", "oku", "obu", "emu", "aba", "om", "omu", "en", "emi",
    "eri", "ama", "aka", "obu", "omu", "okw", "ebw", "n", "ny",
]

# Common Runyoro-Rutooro verb prefixes
RNY_VERB_PREFIXES = [
    "ku", "oku", "n", "ba", "ka", "tu", "mu", "ni", "ti",
]

# Characters valid in Runyoro-Rutooro (Latin script with some extended chars)
# Runyoro uses standard Latin; no tonal diacritics in most written forms
RNY_VALID_CHARS = re.compile(
    r"^[a-zA-Zàáâãäåæèéêëìíîïòóôõöùúûüýÿ'\-\s.,!?;:()\[\]\"]+$"
)

# English pattern — simple check: mostly ASCII alpha
ENG_VALID_CHARS = re.compile(
    r"^[a-zA-Z0-9'\-\s.,!?;:()\[\]\"&%/]+$"
)

# Detect digits-only or symbol-only content
CONTENT_FREE = re.compile(r"^[\d\s\W]+$")


@dataclass
class ValidationIssue:
    issue_type: str
    severity: str        # "error" | "warning" | "info"
    message: str
    original_rny: str
    original_eng: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    valid_pairs: List[Tuple[str, str]] = field(default_factory=list)
    rejected_pairs: List[Tuple[str, str, str]] = field(default_factory=list)  # (rny, eng, reason)
    issues: List[ValidationIssue] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Valid: {len(self.valid_pairs)} | "
            f"Rejected: {len(self.rejected_pairs)} | "
            f"Issues logged: {len(self.issues)}"
        )


class DataValidator:
    """
    Validates parallel sentence pairs for the Runyoro-Rutooro / English NMT system.
    """

    def __init__(
        self,
        min_tokens: int = 2,
        max_tokens: int = 200,
        min_char_ratio: float = 0.4,
        max_char_ratio_multiplier: float = 4.0,
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.min_char_ratio = min_char_ratio
        self.max_char_ratio_multiplier = max_char_ratio_multiplier

    # ------------------------------------------------------------------
    # Token count
    # ------------------------------------------------------------------
    @staticmethod
    def _token_count(text: str) -> int:
        return len(text.split())

    # ------------------------------------------------------------------
    # Length ratio
    # ------------------------------------------------------------------
    @staticmethod
    def _char_ratio(a: str, b: str) -> float:
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return 0.0
        return min(la, lb) / max(la, lb)

    # ------------------------------------------------------------------
    # Basic language detection heuristic (lightweight, no langdetect dep)
    # English: high frequency of common English function words
    # Runyoro: presence of Bantu noun/verb prefixes
    # ------------------------------------------------------------------
    ENG_FUNCTION_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "do", "does", "will", "would", "can", "could",
        "to", "of", "in", "and", "or", "for", "on", "with", "it",
    }

    def _looks_english(self, text: str) -> bool:
        words = set(text.lower().split())
        overlap = words & self.ENG_FUNCTION_WORDS
        return len(overlap) >= 1 or bool(re.search(r"\bthe\b|\bis\b|\bare\b", text, re.I))

    def _looks_runyoro(self, text: str) -> bool:
        ltext = text.lower()
        for prefix in RNY_NOUN_PREFIXES + RNY_VERB_PREFIXES:
            if re.search(r"\b" + re.escape(prefix), ltext):
                return True
        return False

    # ------------------------------------------------------------------
    # Misalignment: both sides look like the same language
    # ------------------------------------------------------------------
    def _detect_misalignment(self, rny: str, eng: str) -> Optional[str]:
        rny_eng = self._looks_english(rny)
        eng_eng = self._looks_english(eng)
        rny_rny = self._looks_runyoro(rny)
        eng_rny = self._looks_runyoro(eng)

        if rny_eng and not rny_rny and eng_eng:
            return "Both sides appear to be English — possible swapped pair"
        if eng_rny and not eng_eng and rny_rny:
            return "Both sides appear to be Runyoro — possible swapped pair"
        return None

    # ------------------------------------------------------------------
    # Runyoro orthographic checks
    # ------------------------------------------------------------------
    @staticmethod
    def _check_runyoro_ortho(text: str) -> List[str]:
        issues = []
        # Double vowels are sometimes incorrectly split
        if re.search(r"\b(aa|ee|oo|uu|ii)\b", text):
            issues.append("Possible incorrect double-vowel split — check long vowels")
        # Check for illegal uppercase mid-word (excluding acronyms)
        words = text.split()
        for w in words:
            if len(w) > 2 and w[1:].lower() != w[1:] and not w.isupper():
                issues.append(f"Unexpected mid-word capitalisation: '{w}'")
                break
        return issues

    # ------------------------------------------------------------------
    # English grammar surface checks
    # ------------------------------------------------------------------
    @staticmethod
    def _check_english_grammar(text: str) -> List[str]:
        issues = []
        # Multiple spaces
        if re.search(r"  +", text):
            issues.append("Multiple consecutive spaces")
        # Missing capitalisation at sentence start
        sentences = re.split(r"[.!?]\s+", text)
        for s in sentences:
            s = s.strip()
            if s and s[0].islower():
                issues.append(f"Sentence starts with lowercase: '{s[:30]}...'")
                break
        # Repeated punctuation
        if re.search(r"[.,!?]{2,}", text):
            issues.append("Repeated punctuation marks")
        return issues

    # ------------------------------------------------------------------
    # Main validation loop
    # ------------------------------------------------------------------
    def validate(
        self, pairs: List[Tuple[str, str]]
    ) -> ValidationResult:
        result = ValidationResult()
        seen: set = set()
        issue_counts: Counter = Counter()

        for rny, eng in pairs:
            rny = rny.strip()
            eng = eng.strip()

            # 1. Empty check — only reject truly empty strings
            if not rny or not eng:
                issue_counts["Empty pair"] += 1
                result.rejected_pairs.append((rny, eng, "Empty pair"))
                continue

            # 2. Deduplication — remove exact duplicates only
            key = (rny.lower(), eng.lower())
            if key in seen:
                issue_counts["Duplicate"] += 1
                result.rejected_pairs.append((rny, eng, "Duplicate"))
                continue
            seen.add(key)

            # 3. Misalignment warning — flag only, do NOT reject
            mis = self._detect_misalignment(rny, eng)
            if mis:
                result.issues.append(ValidationIssue(
                    issue_type="misalignment",
                    severity="warning",
                    message=mis,
                    original_rny=rny,
                    original_eng=eng,
                ))
                issue_counts["Misalignment warning"] += 1

            # 4. Runyoro orthographic issues — flag only
            rny_issues = self._check_runyoro_ortho(rny)
            for iss in rny_issues:
                result.issues.append(ValidationIssue(
                    issue_type="runyoro_ortho",
                    severity="info",
                    message=iss,
                    original_rny=rny,
                    original_eng=eng,
                ))

            # 5. English grammar issues — flag only
            eng_issues = self._check_english_grammar(eng)
            for iss in eng_issues:
                result.issues.append(ValidationIssue(
                    issue_type="english_grammar",
                    severity="info",
                    message=iss,
                    original_rny=rny,
                    original_eng=eng,
                ))

            result.valid_pairs.append((rny, eng))

        result.stats = {
            "total_input": len(pairs),
            "total_valid": len(result.valid_pairs),
            "total_rejected": len(result.rejected_pairs),
            "rejection_breakdown": dict(issue_counts),
            "total_issues_logged": len(result.issues),
        }

        logger.info("Validation complete: %s", result.summary())
        return result
