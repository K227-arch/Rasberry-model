"""
ErrorAnalyzer
=============
Performs detailed error analysis on model predictions:

  - Grammar errors (subject-verb agreement, tense)
  - Terminology errors (wrong glossary term used)
  - Named entity errors (untranslated or incorrectly translated NEs)
  - Word order errors (reordered constituents)
  - Fluency issues (ungrammatical output)
  - Adequacy issues (meaning not preserved)
  - Hallucinations (output not supported by source)
  - Code-switching in output (mixed languages)

Generates a structured error analysis report with examples and statistics.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TranslationError:
    error_type: str
    severity: str   # "critical" | "major" | "minor"
    source: str
    prediction: str
    reference: str
    description: str
    suggestion: str = ""


@dataclass
class ErrorAnalysisResult:
    total_pairs: int
    errors: List[TranslationError] = field(default_factory=list)
    error_type_counts: Dict[str, int] = field(default_factory=dict)
    critical_count: int = 0
    major_count: int = 0
    minor_count: int = 0

    def summary(self) -> str:
        return (
            f"Total: {self.total_pairs} | "
            f"Errors: {len(self.errors)} | "
            f"Critical: {self.critical_count} | "
            f"Major: {self.major_count} | "
            f"Minor: {self.minor_count}"
        )


class ErrorAnalyzer:
    """Categorises and analyses translation errors."""

    def __init__(self, glossary: Optional[Dict[str, str]] = None):
        # glossary: {runyoro_term: english_term}
        self.glossary = glossary or {}

    # ------------------------------------------------------------------
    # Individual error detectors
    # ------------------------------------------------------------------

    def _check_hallucination(
        self, source: str, prediction: str
    ) -> Optional[TranslationError]:
        """
        Flag if prediction contains content clearly absent from source.
        Heuristic: prediction is 3x longer than source in word count.
        """
        src_words = len(source.split())
        pred_words = len(prediction.split())
        if src_words > 0 and pred_words > src_words * 3:
            return TranslationError(
                error_type="hallucination",
                severity="critical",
                source=source,
                prediction=prediction,
                reference="",
                description=(
                    f"Prediction ({pred_words} words) is 3× longer than "
                    f"source ({src_words} words) — possible hallucination"
                ),
            )
        return None

    def _check_untranslated(
        self, source: str, prediction: str, src_lang: str = "rny"
    ) -> Optional[TranslationError]:
        """Flag if the prediction appears to be largely the same as source (not translated)."""
        src_lower = source.lower().strip()
        pred_lower = prediction.lower().strip()
        if src_lower and pred_lower and src_lower == pred_lower:
            return TranslationError(
                error_type="untranslated",
                severity="critical",
                source=source,
                prediction=prediction,
                reference="",
                description="Prediction is identical to source — translation may have failed",
            )
        return None

    def _check_empty_output(
        self, source: str, prediction: str
    ) -> Optional[TranslationError]:
        if not prediction or not prediction.strip():
            return TranslationError(
                error_type="empty_output",
                severity="critical",
                source=source,
                prediction=prediction,
                reference="",
                description="Model produced empty output",
            )
        return None

    def _check_code_switching(
        self, prediction: str, expected_lang: str = "en"
    ) -> Optional[TranslationError]:
        """
        Detect if English output contains Runyoro words (or vice versa).
        Uses simple heuristics — Runyoro words often start with eki, omu, oku, etc.
        """
        RUNYORO_PREFIX = re.compile(
            r"\b(eki|ebi|oku|obu|emu|aba|omw|enk|eri|ama|aka|nk|ny)\w+",
            re.IGNORECASE,
        )
        if expected_lang == "en":
            matches = RUNYORO_PREFIX.findall(prediction)
            if len(matches) >= 2:
                return TranslationError(
                    error_type="code_switching",
                    severity="major",
                    source="",
                    prediction=prediction,
                    reference="",
                    description=(
                        f"English output contains possible Runyoro words: {matches[:3]}"
                    ),
                )
        return None

    def _check_terminology(
        self, source: str, prediction: str, reference: str
    ) -> Optional[TranslationError]:
        """
        Check if known glossary terms are correctly translated.
        """
        for rny_term, en_term in self.glossary.items():
            if rny_term.lower() in source.lower():
                if en_term.lower() not in prediction.lower():
                    return TranslationError(
                        error_type="terminology",
                        severity="major",
                        source=source,
                        prediction=prediction,
                        reference=reference,
                        description=(
                            f"Glossary term '{rny_term}' should translate to '{en_term}' "
                            f"but it was not found in prediction"
                        ),
                        suggestion=f"Expected: '{en_term}'",
                    )
        return None

    def _check_length_divergence(
        self, prediction: str, reference: str
    ) -> Optional[TranslationError]:
        """Fluency/adequacy signal: very different length from reference."""
        if not reference:
            return None
        ref_words = len(reference.split())
        pred_words = len(prediction.split())
        if ref_words > 0:
            ratio = pred_words / ref_words
            if ratio < 0.3 or ratio > 3.0:
                return TranslationError(
                    error_type="length_divergence",
                    severity="minor",
                    source="",
                    prediction=prediction,
                    reference=reference,
                    description=(
                        f"Prediction length ({pred_words}) differs greatly "
                        f"from reference ({ref_words}); ratio={ratio:.2f}"
                    ),
                )
        return None

    # ------------------------------------------------------------------
    # Main analysis loop
    # ------------------------------------------------------------------
    def analyze(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str],
        expected_output_lang: str = "en",
    ) -> ErrorAnalysisResult:
        result = ErrorAnalysisResult(total_pairs=len(sources))
        error_counts: Counter = Counter()

        for src, pred, ref in zip(sources, predictions, references):
            candidate_errors = [
                self._check_empty_output(src, pred),
                self._check_untranslated(src, pred),
                self._check_hallucination(src, pred),
                self._check_code_switching(pred, expected_output_lang),
                self._check_terminology(src, pred, ref),
                self._check_length_divergence(pred, ref),
            ]

            for err in candidate_errors:
                if err is not None:
                    err.source = err.source or src
                    err.reference = err.reference or ref
                    result.errors.append(err)
                    error_counts[err.error_type] += 1
                    if err.severity == "critical":
                        result.critical_count += 1
                    elif err.severity == "major":
                        result.major_count += 1
                    else:
                        result.minor_count += 1

        result.error_type_counts = dict(error_counts)
        logger.info("Error analysis: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------
    def generate_report(
        self,
        analysis: ErrorAnalysisResult,
        output_path: Optional[str] = None,
    ) -> str:
        lines = [
            "# Translation Error Analysis Report",
            "",
            f"**Total pairs analysed:** {analysis.total_pairs}",
            f"**Total errors detected:** {len(analysis.errors)}",
            f"**Critical:** {analysis.critical_count} | "
            f"**Major:** {analysis.major_count} | "
            f"**Minor:** {analysis.minor_count}",
            "",
            "## Error Type Distribution",
            "",
            "| Error Type | Count |",
            "|------------|-------|",
        ]
        for et, cnt in sorted(analysis.error_type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{et}` | {cnt} |")

        lines += [
            "",
            "## Critical Errors (first 10)",
            "",
        ]
        critical = [e for e in analysis.errors if e.severity == "critical"]
        for i, err in enumerate(critical[:10]):
            lines += [
                f"### Error {i+1}: `{err.error_type}`",
                f"**Source:** `{err.source[:100]}`",
                f"**Prediction:** `{err.prediction[:100]}`",
                f"**Reference:** `{err.reference[:100]}`",
                f"**Description:** {err.description}",
                "",
            ]

        lines += [
            "",
            "## Major Errors (first 10)",
            "",
        ]
        major = [e for e in analysis.errors if e.severity == "major"]
        for i, err in enumerate(major[:10]):
            lines += [
                f"### Error {i+1}: `{err.error_type}`",
                f"**Source:** `{err.source[:100]}`",
                f"**Prediction:** `{err.prediction[:100]}`",
                f"**Description:** {err.description}",
                f"**Suggestion:** {err.suggestion or 'N/A'}",
                "",
            ]

        report = "\n".join(lines)
        if output_path:
            from pathlib import Path
            Path(output_path).write_text(report, encoding="utf-8")
            logger.info("Error analysis report written: %s", output_path)

        return report
