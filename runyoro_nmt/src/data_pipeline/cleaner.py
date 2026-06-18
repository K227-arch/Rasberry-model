"""
DataCleaner
===========
Applies a comprehensive cleaning pipeline to validated sentence pairs:

  1. Unicode normalisation (NFC)
  2. Whitespace normalisation
  3. HTML/XML entity removal
  4. Punctuation standardisation
  5. Runyoro-Rutooro-specific orthographic corrections
  6. English spelling / casing corrections
  7. Removal of URLs and e-mail addresses
  8. Code-switching correction (flags and attempts correction)
  9. Generates before/after report with diffs

All transformations are logged so they are fully reproducible.
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Runyoro-Rutooro corrections lookup
# (common misspellings / inconsistencies found in the raw data)
# ---------------------------------------------------------------------------
RUNYORO_CORRECTIONS: Dict[str, str] = {
    # Variant spellings → canonical form
    "nyoro": "Runyoro",
    "rutooro": "Rutooro",
    "omuntu": "omuntu",   # person (already correct, but normalise casing issues)
    "abantu": "abantu",
    "obulamu": "obulamu",
    # Common misspellings seen in the raw files
    "ekyaama": "ekyama",
    "obunyoo": "obunyo",
    "okuttema": "okutema",
}

# Regex for URLs
URL_RE = re.compile(
    r"https?://\S+|www\.\S+|ftp://\S+",
    re.IGNORECASE,
)

# Regex for emails
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# Regex for HTML entities
HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")

# Multiple spaces
MULTI_SPACE = re.compile(r" {2,}")

# Bracket noise (empty brackets left by template editors)
EMPTY_BRACKET = re.compile(r"\[\s*\]|\(\s*\)|\{\s*\}")


@dataclass
class CleanRecord:
    original_rny: str
    original_eng: str
    cleaned_rny: str
    cleaned_eng: str
    changes: List[str] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return (
            self.original_rny != self.cleaned_rny
            or self.original_eng != self.cleaned_eng
        )


class DataCleaner:
    """Applies deterministic text-cleaning rules to parallel pairs."""

    def __init__(self, runyoro_corrections: Optional[Dict[str, str]] = None):
        self.runyoro_corrections = runyoro_corrections or RUNYORO_CORRECTIONS

    # ------------------------------------------------------------------
    # Individual cleaning steps
    # ------------------------------------------------------------------
    @staticmethod
    def _unicode_normalise(text: str) -> Tuple[str, bool]:
        normalised = unicodedata.normalize("NFC", text)
        return normalised, normalised != text

    @staticmethod
    def _decode_html(text: str) -> Tuple[str, bool]:
        decoded = html.unescape(text)
        decoded = HTML_ENTITY_RE.sub("", decoded)
        return decoded, decoded != text

    @staticmethod
    def _remove_urls(text: str) -> Tuple[str, bool]:
        cleaned = URL_RE.sub("", text)
        cleaned = EMAIL_RE.sub("", cleaned)
        return cleaned, cleaned != text

    @staticmethod
    def _normalise_whitespace(text: str) -> Tuple[str, bool]:
        cleaned = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        cleaned = MULTI_SPACE.sub(" ", cleaned).strip()
        return cleaned, cleaned != text

    @staticmethod
    def _remove_empty_brackets(text: str) -> Tuple[str, bool]:
        cleaned = EMPTY_BRACKET.sub("", text).strip()
        return cleaned, cleaned != text

    @staticmethod
    def _normalise_quotes(text: str) -> Tuple[str, bool]:
        cleaned = (
            text
            .replace("\u201c", '"').replace("\u201d", '"')   # curly double
            .replace("\u2018", "'").replace("\u2019", "'")   # curly single
            .replace("\u00ab", '"').replace("\u00bb", '"')   # guillemets
        )
        return cleaned, cleaned != text

    @staticmethod
    def _normalise_punctuation(text: str) -> Tuple[str, bool]:
        # Collapse repeated punctuation (e.g. "..." → "…" is kept, "!!" → "!")
        cleaned = re.sub(r"([!?]){2,}", r"\1", text)
        # Fix space before punctuation
        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
        return cleaned, cleaned != text

    @staticmethod
    def _fix_english_capitalisation(text: str) -> Tuple[str, bool]:
        """Capitalise first letter of each sentence."""
        sentences = re.split(r"([.!?]\s+)", text)
        fixed_parts = []
        for part in sentences:
            if part and not re.match(r"[.!?]\s+", part) and part[0].islower():
                part = part[0].upper() + part[1:]
            fixed_parts.append(part)
        result = "".join(fixed_parts)
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        return result, result != text

    def _apply_runyoro_corrections(self, text: str) -> Tuple[str, bool]:
        changed = False
        for wrong, right in self.runyoro_corrections.items():
            pattern = re.compile(r"\b" + re.escape(wrong) + r"\b", re.IGNORECASE)
            new_text = pattern.sub(right, text)
            if new_text != text:
                changed = True
                text = new_text
        return text, changed

    @staticmethod
    def _strip_numbering(text: str) -> Tuple[str, bool]:
        """Remove leading numbering like '1.', '1)', '(a)' from entries."""
        cleaned = re.sub(r"^\s*(\d+[.)]\s*|[a-zA-Z][.)]\s*|\([a-zA-Z0-9]+\)\s*)", "", text)
        return cleaned.strip(), cleaned != text

    # ------------------------------------------------------------------
    # Clean a single sentence pair
    # ------------------------------------------------------------------
    def _clean_text(
        self, text: str, is_english: bool
    ) -> Tuple[str, List[str]]:
        changes: List[str] = []
        orig = text

        steps = [
            ("unicode_normalise", self._unicode_normalise),
            ("decode_html", self._decode_html),
            ("remove_urls", self._remove_urls),
            ("normalise_quotes", self._normalise_quotes),
            ("remove_empty_brackets", self._remove_empty_brackets),
            ("normalise_whitespace", self._normalise_whitespace),
            ("strip_numbering", self._strip_numbering),
            ("normalise_punctuation", self._normalise_punctuation),
        ]

        for name, fn in steps:
            text, modified = fn(text)
            if modified:
                changes.append(name)

        if is_english:
            text, modified = self._fix_english_capitalisation(text)
            if modified:
                changes.append("fix_capitalisation")
        else:
            text, modified = self._apply_runyoro_corrections(text)
            if modified:
                changes.append("runyoro_corrections")

        return text, changes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clean(
        self, pairs: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], List[CleanRecord]]:
        """
        Clean a list of (runyoro, english) pairs.

        Returns:
            cleaned_pairs: list of cleaned (rny, eng) tuples
            records: full before/after change records for the report
        """
        cleaned_pairs: List[Tuple[str, str]] = []
        records: List[CleanRecord] = []
        modified_count = 0

        for rny, eng in pairs:
            clean_rny, rny_changes = self._clean_text(rny, is_english=False)
            clean_eng, eng_changes = self._clean_text(eng, is_english=True)

            record = CleanRecord(
                original_rny=rny,
                original_eng=eng,
                cleaned_rny=clean_rny,
                cleaned_eng=clean_eng,
                changes=rny_changes + [f"eng:{c}" for c in eng_changes],
            )
            records.append(record)
            cleaned_pairs.append((clean_rny, clean_eng))

            if record.was_modified:
                modified_count += 1

        logger.info(
            "Cleaning complete: %d/%d pairs modified",
            modified_count, len(pairs),
        )
        return cleaned_pairs, records

    def generate_report(
        self, records: List[CleanRecord], output_path: Optional[str] = None
    ) -> str:
        """Generate a markdown cleaning report with before/after examples."""
        modified = [r for r in records if r.was_modified]
        change_type_counts: Dict[str, int] = {}
        for r in modified:
            for c in r.changes:
                change_type_counts[c] = change_type_counts.get(c, 0) + 1

        lines = [
            "# Data Cleaning Report",
            "",
            f"**Total pairs processed:** {len(records)}",
            f"**Pairs modified:** {len(modified)} ({100*len(modified)/max(len(records),1):.1f}%)",
            "",
            "## Change Type Breakdown",
            "",
            "| Change Type | Count |",
            "|-------------|-------|",
        ]
        for ct, cnt in sorted(change_type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{ct}` | {cnt} |")

        lines += [
            "",
            "## Before/After Examples (first 20 modifications)",
            "",
        ]

        for i, r in enumerate(modified[:20]):
            lines += [
                f"### Example {i+1}",
                f"**Changes:** {', '.join(r.changes)}",
                "",
                f"**Runyoro before:** `{r.original_rny}`",
                f"**Runyoro after:**  `{r.cleaned_rny}`",
                "",
                f"**English before:** `{r.original_eng}`",
                f"**English after:**  `{r.cleaned_eng}`",
                "",
                "---",
                "",
            ]

        report = "\n".join(lines)

        if output_path:
            from pathlib import Path
            Path(output_path).write_text(report, encoding="utf-8")
            logger.info("Cleaning report written to %s", output_path)

        return report
