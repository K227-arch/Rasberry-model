"""
TranslationMemoryBuilder
========================
Creates and manages linguistic resources:

  1. Translation Memory (TM)    — reusable sentence-level matches in TMX format
  2. Terminology Database       — term-level glossary in TBX format
  3. Bilingual Glossary         — CSV/JSON export for UI lookup
  4. Named Entity Registry      — proper nouns, places, people
  5. Idiomatic Expression Bank  — culturally grounded phrase pairs

All resources are versioned and cross-referenced.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TMEntry:
    runyoro: str
    english: str
    domain: str = "general"
    source_file: str = ""
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    usage_count: int = 0


@dataclass
class TermEntry:
    runyoro_term: str
    english_term: str
    part_of_speech: str = ""
    domain: str = "general"
    definition_en: str = ""
    definition_rny: str = ""
    synonyms_en: List[str] = field(default_factory=list)
    synonyms_rny: List[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Domain detector (lightweight keyword heuristic)
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "agriculture": [
        "seed", "crop", "soil", "harvest", "farm", "plant", "fertilizer",
        "irrigation", "livestock", "maize", "bean", "rice", "wheat",
        "ekibaro", "okukolera", "omugenyi", "ensigo",
    ],
    "health": [
        "health", "disease", "medicine", "hospital", "doctor", "nurse",
        "obulamu", "omusawo", "ekirwaro",
    ],
    "education": [
        "school", "student", "teacher", "learn", "study", "class",
        "ekyooro", "omwigisha", "omusomi",
    ],
    "legal": [
        "law", "court", "justice", "right", "government", "policy",
        "omutekateka", "etegeko",
    ],
}


def _detect_domain(rny: str, eng: str) -> str:
    combined = (rny + " " + eng).lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return domain
    return "general"


# ---------------------------------------------------------------------------
# TMX serialisation
# ---------------------------------------------------------------------------

def _to_tmx(entries: List[TMEntry]) -> str:
    header = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE tmx SYSTEM "tmx14.dtd">
<tmx version="1.4">
  <header
    creationtool="runyoro-nmt-v1"
    creationtoolversion="1.0"
    datatype="PlainText"
    segtype="sentence"
    adminlang="en"
    srclang="nyk"
    o-tmf="TMX"
  />
  <body>"""

    tus = []
    for e in entries:
        tu = f"""    <tu>
      <tuv xml:lang="nyk"><seg>{_xml_escape(e.runyoro)}</seg></tuv>
      <tuv xml:lang="en"><seg>{_xml_escape(e.english)}</seg></tuv>
    </tu>"""
        tus.append(tu)

    footer = "  </body>\n</tmx>"
    return header + "\n" + "\n".join(tus) + "\n" + footer


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# TBX serialisation
# ---------------------------------------------------------------------------

def _to_tbx(terms: List[TermEntry]) -> str:
    entries = []
    for i, t in enumerate(terms):
        entry = f"""  <termEntry id="te{i:05d}">
    <langSet xml:lang="en">
      <tig>
        <term>{_xml_escape(t.english_term)}</term>
        <termNote type="partOfSpeech">{t.part_of_speech}</termNote>
        <descrip type="definition">{_xml_escape(t.definition_en)}</descrip>
        <termNote type="domain">{t.domain}</termNote>
      </tig>
    </langSet>
    <langSet xml:lang="nyk">
      <tig>
        <term>{_xml_escape(t.runyoro_term)}</term>
        <descrip type="definition">{_xml_escape(t.definition_rny)}</descrip>
      </tig>
    </langSet>
  </termEntry>"""
        entries.append(entry)

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<martif type="TBX" xml:lang="en">\n'
        "  <martifHeader><fileDesc><titleStmt>"
        "<title>Runyoro-Rutooro / English Termbase</title>"
        "</titleStmt></fileDesc></martifHeader>\n"
        "  <text><body>\n"
        + "\n".join(entries)
        + "\n  </body></text>\n</martif>"
    )


# ---------------------------------------------------------------------------
# Named entity extraction (lightweight heuristic)
# ---------------------------------------------------------------------------

def _extract_named_entities(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Extract likely proper-noun pairs using capitalisation heuristics.
    Returns (runyoro_entity, english_entity) pairs.
    """
    entities = []
    for rny, eng in pairs:
        rny_caps = re.findall(r"\b[A-Z][a-z]+\b", rny)
        eng_caps = re.findall(r"\b[A-Z][a-z]+\b", eng)
        if rny_caps and eng_caps:
            for rw, ew in zip(rny_caps, eng_caps):
                if rw != ew:  # not the same word
                    entities.append((rw, ew))
    return list(set(entities))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TranslationMemoryBuilder:
    """Build and export translation memories and terminology resources."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_tm(
        self, pairs: List[Tuple[str, str]], source_file: str = ""
    ) -> List[TMEntry]:
        entries = []
        for rny, eng in pairs:
            domain = _detect_domain(rny, eng)
            entries.append(
                TMEntry(
                    runyoro=rny,
                    english=eng,
                    domain=domain,
                    source_file=source_file,
                )
            )
        logger.info("Built TM with %d entries", len(entries))
        return entries

    def save_tmx(self, entries: List[TMEntry], filename: str = "runyoro_en.tmx") -> Path:
        path = self.output_dir / filename
        path.write_text(_to_tmx(entries), encoding="utf-8")
        logger.info("TMX saved: %s", path)
        return path

    def build_glossary(
        self, pairs: List[Tuple[str, str]]
    ) -> List[TermEntry]:
        """
        Extract single-word / short-phrase term pairs to build a terminology database.
        """
        terms: List[TermEntry] = []
        seen: set = set()
        for rny, eng in pairs:
            # Short pairs (1-3 tokens) are likely vocabulary items
            if len(rny.split()) <= 3 and len(eng.split()) <= 3:
                key = (rny.lower(), eng.lower())
                if key in seen:
                    continue
                seen.add(key)
                domain = _detect_domain(rny, eng)
                terms.append(
                    TermEntry(
                        runyoro_term=rny,
                        english_term=eng,
                        domain=domain,
                    )
                )
        logger.info("Glossary built with %d terms", len(terms))
        return terms

    def save_glossary_csv(
        self, terms: List[TermEntry], filename: str = "glossary.csv"
    ) -> Path:
        path = self.output_dir / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "runyoro_term", "english_term", "part_of_speech",
                    "domain", "definition_en", "notes"
                ],
            )
            writer.writeheader()
            for t in terms:
                writer.writerow({
                    "runyoro_term": t.runyoro_term,
                    "english_term": t.english_term,
                    "part_of_speech": t.part_of_speech,
                    "domain": t.domain,
                    "definition_en": t.definition_en,
                    "notes": t.notes,
                })
        logger.info("Glossary CSV saved: %s", path)
        return path

    def save_glossary_json(
        self, terms: List[TermEntry], filename: str = "glossary.json"
    ) -> Path:
        path = self.output_dir / filename
        data = [
            {
                "runyoro": t.runyoro_term,
                "english": t.english_term,
                "pos": t.part_of_speech,
                "domain": t.domain,
                "definition_en": t.definition_en,
            }
            for t in terms
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Glossary JSON saved: %s", path)
        return path

    def save_tbx(
        self, terms: List[TermEntry], filename: str = "runyoro_en.tbx"
    ) -> Path:
        path = self.output_dir / filename
        path.write_text(_to_tbx(terms), encoding="utf-8")
        logger.info("TBX saved: %s", path)
        return path

    def save_named_entities(
        self, pairs: List[Tuple[str, str]], filename: str = "named_entities.json"
    ) -> Path:
        entities = _extract_named_entities(pairs)
        path = self.output_dir / filename
        data = [{"runyoro": r, "english": e} for r, e in entities]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Named entities saved: %d entries -> %s", len(entities), path)
        return path

    def build_all(self, pairs: List[Tuple[str, str]]) -> Dict[str, Path]:
        """Run the full TM + glossary + TBX pipeline and save all outputs."""
        logger.info("Building all linguistic resources from %d pairs...", len(pairs))

        tm_entries = self.build_tm(pairs)
        glossary_terms = self.build_glossary(pairs)

        paths = {
            "tmx": self.save_tmx(tm_entries),
            "glossary_csv": self.save_glossary_csv(glossary_terms),
            "glossary_json": self.save_glossary_json(glossary_terms),
            "tbx": self.save_tbx(glossary_terms),
            "named_entities": self.save_named_entities(pairs),
        }

        logger.info("All linguistic resources saved to %s", self.output_dir)
        return paths
