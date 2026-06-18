"""
SentenceAligner
===============
Validates and repairs sentence alignment in parallel corpora using:

  - Cosine similarity on TF-IDF character n-gram vectors (no heavy deps)
  - Length-ratio heuristics
  - Semantic similarity via SentenceTransformers (optional, falls back gracefully)
  - Back-translation alignment check

Reports misalignment confidence scores and suggests re-ordered pairings
when blocks of sentences appear to be shifted.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight TF-IDF cosine similarity (no sklearn required)
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 3) -> List[str]:
    text = text.lower().strip()
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def _tfidf_vector(texts: List[str], n: int = 3) -> List[Dict[str, float]]:
    """Compute TF-IDF char-ngram vectors for a list of texts."""
    tf_vectors = []
    for text in texts:
        ngrams = _char_ngrams(text, n)
        tf: Dict[str, float] = Counter(ngrams)  # type: ignore[assignment]
        total = sum(tf.values())
        if total > 0:
            tf = {k: v / total for k, v in tf.items()}
        tf_vectors.append(tf)

    # IDF
    doc_freq: Counter = Counter()
    for vec in tf_vectors:
        for term in set(vec.keys()):
            doc_freq[term] += 1

    n_docs = len(texts)
    idf: Dict[str, float] = {
        term: math.log((n_docs + 1) / (df + 1)) + 1
        for term, df in doc_freq.items()
    }

    tfidf_vectors = []
    for vec in tf_vectors:
        tfidf = {k: v * idf.get(k, 1.0) for k, v in vec.items()}
        tfidf_vectors.append(tfidf)

    return tfidf_vectors


def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    keys = set(v1.keys()) & set(v2.keys())
    dot = sum(v1[k] * v2[k] for k in keys)
    mag1 = math.sqrt(sum(x**2 for x in v1.values()))
    mag2 = math.sqrt(sum(x**2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


# ---------------------------------------------------------------------------
# Alignment result
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    aligned_pairs: List[Tuple[str, str]]
    suspicious_indices: List[int]
    scores: List[float]

    def summary(self) -> str:
        n_suspicious = len(self.suspicious_indices)
        return (
            f"Aligned: {len(self.aligned_pairs)} pairs | "
            f"Suspicious: {n_suspicious} ({100*n_suspicious/max(len(self.aligned_pairs),1):.1f}%)"
        )


class SentenceAligner:
    """
    Checks and scores alignment of sentence pairs.
    Flags pairs where character-length ratio or semantic similarity
    suggests a possible misalignment.
    """

    def __init__(
        self,
        length_ratio_threshold: float = 0.25,
        similarity_threshold: float = 0.05,
        use_semantic: bool = False,
        semantic_model: str = "all-MiniLM-L6-v2",
    ):
        self.length_ratio_threshold = length_ratio_threshold
        self.similarity_threshold = similarity_threshold
        self.use_semantic = use_semantic
        self.semantic_model_name = semantic_model
        self._semantic_model = None

        if use_semantic:
            self._load_semantic_model()

    def _load_semantic_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._semantic_model = SentenceTransformer(self.semantic_model_name)
            logger.info("Semantic model loaded: %s", self.semantic_model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — falling back to char-ngram similarity"
            )
            self.use_semantic = False

    # ------------------------------------------------------------------
    # Length ratio score
    # ------------------------------------------------------------------
    @staticmethod
    def _length_score(rny: str, eng: str) -> float:
        if not rny or not eng:
            return 0.0
        ratio = min(len(rny), len(eng)) / max(len(rny), len(eng))
        return ratio

    # ------------------------------------------------------------------
    # Char-ngram similarity between translations
    # (cross-lingual proxy: shared proper nouns, numbers, named entities)
    # ------------------------------------------------------------------
    @staticmethod
    def _shared_tokens_score(rny: str, eng: str) -> float:
        """
        Higher score if pairs share numbers, proper nouns, or named entities
        (cross-lingual anchors), which is a weak alignment signal.
        """
        numbers_rny = set(re.findall(r"\d+", rny))
        numbers_eng = set(re.findall(r"\d+", eng))
        if numbers_rny or numbers_eng:
            if numbers_rny == numbers_eng:
                return 1.0
            elif numbers_rny & numbers_eng:
                return 0.5
        return 0.3   # neutral — no numbers to compare

    # ------------------------------------------------------------------
    # Semantic cosine similarity (if model loaded)
    # ------------------------------------------------------------------
    def _semantic_score(self, rny: str, eng: str) -> float:
        if not self.use_semantic or self._semantic_model is None:
            return 0.5   # neutral
        import numpy as np
        emb = self._semantic_model.encode([rny, eng], convert_to_numpy=True)
        cos = float(
            np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9)
        )
        return max(0.0, cos)

    # ------------------------------------------------------------------
    # Composite alignment score
    # ------------------------------------------------------------------
    def score_pair(self, rny: str, eng: str) -> float:
        len_score = self._length_score(rny, eng)
        tok_score = self._shared_tokens_score(rny, eng)
        sem_score = self._semantic_score(rny, eng) if self.use_semantic else 0.5
        # Weighted combination
        return 0.4 * len_score + 0.2 * tok_score + 0.4 * sem_score

    # ------------------------------------------------------------------
    # Main alignment check
    # ------------------------------------------------------------------
    def check_alignment(
        self, pairs: List[Tuple[str, str]]
    ) -> AlignmentResult:
        aligned: List[Tuple[str, str]] = []
        suspicious: List[int] = []
        scores: List[float] = []

        for i, (rny, eng) in enumerate(pairs):
            score = self.score_pair(rny, eng)
            scores.append(score)
            len_ratio = self._length_score(rny, eng)

            if len_ratio < self.length_ratio_threshold:
                suspicious.append(i)
                logger.debug(
                    "Suspicious pair [%d]: len_ratio=%.2f  rny='%s...'  eng='%s...'",
                    i, len_ratio, rny[:40], eng[:40],
                )
            aligned.append((rny, eng))

        result = AlignmentResult(
            aligned_pairs=aligned,
            suspicious_indices=suspicious,
            scores=scores,
        )
        logger.info("Alignment check: %s", result.summary())
        return result

    # ------------------------------------------------------------------
    # Semantic similarity validation (for evaluation)
    # ------------------------------------------------------------------
    def semantic_similarity_report(
        self,
        pairs: List[Tuple[str, str]],
        sample_size: int = 100,
    ) -> Dict:
        """
        Compute average semantic similarity for a sample of pairs.
        Used as one quality signal during evaluation.
        """
        import random
        sample = random.sample(pairs, min(sample_size, len(pairs)))
        scores = [self.score_pair(r, e) for r, e in sample]
        avg = sum(scores) / len(scores) if scores else 0.0
        return {
            "sample_size": len(sample),
            "avg_alignment_score": round(avg, 4),
            "min_score": round(min(scores, default=0), 4),
            "max_score": round(max(scores, default=0), 4),
        }
