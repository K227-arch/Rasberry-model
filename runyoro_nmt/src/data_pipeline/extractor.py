"""
DataExtractor
=============
Reads all raw data files (xlsx, docx, csv) from the `raw data/` folder.

DOCX dictionary files have this column structure:
  - Word           = Runyoro-Rutooro headword  (col 0)
  - Definition     = English meaning           (col 5 or 6)
  - Example (Rny)  = Runyoro example sentence  (second-to-last col)
  - Example (Eng)  = English example sentence  (last col)

Extraction strategy:
  1. Word  <->  Definition  (always extracted — single words included)
  2. Example(Rny) <-> Example(Eng)  — only when the Runyoro example
     is actually in Runyoro and the English example is actually in English.
     Examples in the wrong language are DROPPED.

XLSX / CSV files: column pair auto-detected by heuristic name matching.

No minimum token length filter — single words are valuable training pairs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-name heuristics for XLSX / CSV
# ---------------------------------------------------------------------------
RUNYORO_HINTS = [
    "runyoro", "rutooro", "nyoro", "runya", "nyk", "rutoro",
    "local", "vernacular", "native",
]
ENGLISH_HINTS = [
    "english", "eng", "en", "translation", "meaning", "gloss",
    "definition", "equiv",
]

# English function words — used to detect if a "Runyoro" cell is actually English
ENGLISH_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "do", "does", "will", "would", "can", "could",
    "to", "of", "in", "and", "or", "for", "on", "with", "it",
    "he", "she", "they", "we", "i", "you", "his", "her", "their",
    "this", "that", "not", "but", "so", "if", "when", "by", "at",
    "from", "as", "up", "out", "about", "into", "through",
}

# Runyoro-Rutooro Bantu prefix markers
RUNYORO_PREFIXES = re.compile(
    r"\b(eki|ebi|oku|obu|emu|aba|omw|enk|eri|ama|aka|omu|ku|tu|ba|ni|"
    r"ha|ka|mu|na|ng|ny|by|gy|ky|my|ry|sy|hy)\w",
    re.IGNORECASE,
)


def _is_english(text: str, threshold: float = 0.35) -> bool:
    """Return True if text looks predominantly English."""
    if not text:
        return False
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return False
    eng_count = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    return (eng_count / len(words)) >= threshold


def _is_runyoro(text: str) -> bool:
    """Return True if text has Runyoro-Rutooro markers."""
    if not text:
        return False
    return bool(RUNYORO_PREFIXES.search(text))


def _likely_runyoro_not_english(text: str) -> bool:
    """
    Returns True if text is suitable as a Runyoro-Rutooro side:
    - Not clearly English (low English function-word ratio)
    - Optionally has Runyoro prefixes
    """
    if not text or not text.strip():
        return False
    # Pure numbers/symbols — skip
    if re.fullmatch(r"[\d\s\W]+", text):
        return False
    # If it looks like English, skip
    if _is_english(text, threshold=0.30):
        return False
    return True


def _likely_english_not_runyoro(text: str) -> bool:
    """
    Returns True if text is suitable as an English side:
    - Has some English content
    - Not clearly a Runyoro sentence (no heavy Bantu prefix load)
    """
    if not text or not text.strip():
        return False
    if re.fullmatch(r"[\d\s\W]+", text):
        return False
    # Strong Runyoro signals — skip
    words = re.findall(r"[a-zA-Z]+", text)
    if not words:
        return False
    rny_hits = sum(1 for w in words if RUNYORO_PREFIXES.match(w))
    if len(words) > 0 and (rny_hits / len(words)) > 0.5:
        return False
    return True


def _clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Column-name scoring helper
# ---------------------------------------------------------------------------
def _score_column(name: str, hints: List[str]) -> int:
    n = name.lower().strip()
    for i, h in enumerate(hints):
        if h in n:
            return len(hints) - i
    return 0


def _find_pair(columns: List[str]) -> Optional[Tuple[str, str]]:
    """Return (runyoro_col, english_col) with best heuristic score."""
    rny = sorted(columns, key=lambda c: _score_column(c, RUNYORO_HINTS), reverse=True)
    eng = sorted(columns, key=lambda c: _score_column(c, ENGLISH_HINTS), reverse=True)
    rny_best = rny[0] if _score_column(rny[0], RUNYORO_HINTS) > 0 else None
    eng_best = eng[0] if _score_column(eng[0], ENGLISH_HINTS) > 0 else None
    if rny_best and eng_best and rny_best != eng_best:
        return rny_best, eng_best
    if len(columns) == 2:
        logger.warning("Column heuristic fallback: %s assumed (runyoro, english)", columns)
        return columns[0], columns[1]
    return None


# ---------------------------------------------------------------------------
# Excel / CSV  — NOTE: no minimum token filter, single words included
# ---------------------------------------------------------------------------
def _extract_spreadsheet(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    if path.suffix.lower() == ".csv":
        sheets = {"sheet1": pd.read_csv(path, dtype=str)}
    else:
        xf = pd.ExcelFile(path)
        sheets = {s: xf.parse(s, dtype=str) for s in xf.sheet_names}

    for sheet_name, df in sheets.items():
        df.columns = [str(c).strip() for c in df.columns]
        result = _find_pair(list(df.columns))
        if not result:
            logger.warning("  [%s/%s] No column pair — skipping", path.name, sheet_name)
            continue
        rny_col, eng_col = result

        # Detect which is really Runyoro vs English by sampling
        # (augmentted pos pairs has 'english' and 'rutoro' but 'english' col
        #  is actually the ENGLISH side and 'rutoro' is the Runyoro side)
        sample_rny = df[rny_col].dropna().head(20).astype(str).tolist()
        sample_eng = df[eng_col].dropna().head(20).astype(str).tolist()
        rny_looks_eng = sum(_is_english(t) for t in sample_rny)
        eng_looks_eng = sum(_is_english(t) for t in sample_eng)

        # Swap if the "rny" column actually looks more English than the "eng" col
        if rny_looks_eng > eng_looks_eng:
            rny_col, eng_col = eng_col, rny_col
            logger.info(
                "  [%s/%s] Swapped columns: Runyoro='%s'  English='%s'",
                path.name, sheet_name, rny_col, eng_col,
            )
        else:
            logger.info(
                "  [%s/%s] Columns: Runyoro='%s'  English='%s'",
                path.name, sheet_name, rny_col, eng_col,
            )

        for _, row in df.iterrows():
            rny = str(row.get(rny_col, "")).strip()
            eng = str(row.get(eng_col, "")).strip()
            if not rny or not eng or rny == "nan" or eng == "nan":
                continue
            # Only drop if content-free (pure symbols/numbers)
            if re.fullmatch(r"[\d\s\W]+", rny) or re.fullmatch(r"[\d\s\W]+", eng):
                continue
            # Strip POS category tags from augmented pos pairs xlsx
            # e.g. "[GENERAL_NOUN] someone's relatives" -> "someone's relatives"
            rny = re.sub(r"^\[[A-Z_]+\]\s*", "", rny).strip()
            eng = re.sub(r"^\[[A-Z_]+\]\s*", "", eng).strip()
            if not rny or not eng:
                continue
            pairs.append((rny, eng))

    logger.info("  [%s] %d pairs from spreadsheet", path.name, len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# DOCX — Dictionary files
# ---------------------------------------------------------------------------
# Expected column patterns in the dictionary docx files:
#   col 0  : Word (Runyoro headword)
#   col 5/6: Definition (English meaning)  ← "Definition" in header
#   col -2 : Example (Runyoro)
#   col -1 : Example (English)

def _find_col_idx(headers: List[str], keywords: List[str]) -> int:
    """Return index of first header matching any keyword (case-insensitive)."""
    for kw in keywords:
        for i, h in enumerate(headers):
            if kw.lower() in h.lower():
                return i
    return -1


def _extract_docx(path: Path) -> List[Tuple[str, str]]:
    """
    Extract pairs from dictionary .docx files.

    Two kinds of pairs per row:
      1. Word <-> Definition   (direct translation — kept even for single words)
      2. Example(Runyoro) <-> Example(English)
         — dropped if Runyoro example is actually in English
         — dropped if English example is actually in Runyoro
    """
    try:
        from docx import Document  # type: ignore
    except ImportError:
        logger.error("python-docx not installed.")
        return []

    doc = Document(str(path))
    pairs: List[Tuple[str, str]] = []
    dropped_lang = 0

    for table in doc.tables:
        if not table.rows:
            continue

        headers = [_clean_cell(c.text) for c in table.rows[0].cells]

        word_idx   = 0
        def_idx    = _find_col_idx(headers, ["definition"])
        ex_rny_idx = _find_col_idx(headers, [
            "example (runyoro)", "example runyoro",
            "runyoro example", "example(runyoro)",
        ])
        ex_eng_idx = _find_col_idx(headers, [
            "example (english)", "example english",
            "english example", "example(english)",
        ])

        # Fallback: scan from right for example columns
        if ex_rny_idx < 0 or ex_eng_idx < 0:
            for i in range(len(headers) - 1, -1, -1):
                h = headers[i].lower()
                if "english" in h and ex_eng_idx < 0:
                    ex_eng_idx = i
                elif ("runyoro" in h or "rutooro" in h) and ex_rny_idx < 0:
                    ex_rny_idx = i
                if ex_rny_idx >= 0 and ex_eng_idx >= 0:
                    break

        for row in table.rows[1:]:
            cells = row.cells
            n = len(cells)

            # ── 1. Word ↔ Definition ─────────────────────────────────
            word = _clean_cell(cells[word_idx].text) if word_idx < n else ""
            defn = _clean_cell(cells[def_idx].text) if 0 <= def_idx < n else ""

            # Skip letter/alphabet intro rows
            if (word and defn and len(word) < 120
                    and not re.match(r"^[A-Za-z]\s+[A-Za-z]{2}", word)):
                word_ok = _likely_runyoro_not_english(word) or (
                    len(word.split()) == 1 and word.isalpha()
                )
                defn_ok = _likely_english_not_runyoro(defn) and len(defn) > 1
                if word_ok and defn_ok:
                    pairs.append((word, defn))

            # ── 2. Example (Runyoro) ↔ Example (English) ─────────────
            if ex_rny_idx >= 0 and ex_eng_idx >= 0:
                ex_rny = _clean_cell(cells[ex_rny_idx].text) if ex_rny_idx < n else ""
                ex_eng = _clean_cell(cells[ex_eng_idx].text) if ex_eng_idx < n else ""

                if ex_rny and ex_eng and ex_rny not in ("-","") and ex_eng not in ("-",""):
                    rny_parts = [p.strip() for p in re.split(r"\n+", ex_rny) if p.strip()]
                    eng_parts = [p.strip() for p in re.split(r"\n+", ex_eng) if p.strip()]

                    for rp, ep in zip(rny_parts, eng_parts):
                        rp = _clean_cell(rp)
                        ep = _clean_cell(ep)
                        if not rp or not ep or rp in ("-","--") or ep in ("-","--"):
                            continue
                        if _likely_runyoro_not_english(rp) and _likely_english_not_runyoro(ep):
                            pairs.append((rp, ep))
                        else:
                            dropped_lang += 1

    logger.info(
        "  [%s] %d pairs | %d examples dropped (wrong language)",
        path.name, len(pairs), dropped_lang,
    )
    return pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class DataExtractor:
    """Extract parallel sentence pairs from raw data files."""

    def __init__(self, raw_data_dir: str | Path):
        self.raw_data_dir = Path(raw_data_dir)
        self._supported = {".xlsx", ".xls", ".csv", ".docx"}

    def extract_all(self) -> Dict[str, List[Tuple[str, str]]]:
        results: Dict[str, List[Tuple[str, str]]] = {}
        if not self.raw_data_dir.exists():
            logger.error("Raw data directory not found: %s", self.raw_data_dir)
            return results

        files = [f for f in self.raw_data_dir.iterdir()
                 if f.suffix.lower() in self._supported]
        if not files:
            logger.warning("No supported files found in %s", self.raw_data_dir)
            return results

        logger.info("Found %d raw data files", len(files))
        for path in sorted(files):
            logger.info("Extracting: %s", path.name)
            try:
                if path.suffix.lower() in {".xlsx", ".xls", ".csv"}:
                    pairs = _extract_spreadsheet(path)
                elif path.suffix.lower() == ".docx":
                    pairs = _extract_docx(path)
                else:
                    continue
                logger.info("  -> %d pairs", len(pairs))
                results[path.name] = pairs
            except Exception as exc:
                logger.error("  Error: %s: %s", path.name, exc, exc_info=True)

        total = sum(len(v) for v in results.values())
        logger.info("Total raw pairs extracted: %d", total)
        return results

    def extract_flat(self) -> List[Tuple[str, str]]:
        all_pairs = []
        for pairs in self.extract_all().values():
            all_pairs.extend(pairs)
        return all_pairs
