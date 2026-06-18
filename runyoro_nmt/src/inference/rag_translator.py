"""
RAGTranslator
=============
Retrieval-Augmented Translation (RAT):
Retrieves similar previously-translated sentence pairs from the
translation memory (FAISS index) and injects them as few-shot
examples into the translation prompt — improving accuracy for
domain-specific terms and idiomatic expressions.

Architecture matches the Raspberry Pi RAG design:
  - FAISS CPU index for fast retrieval
  - all-MiniLM-L6-v2 for embedding (lightweight)
  - Top-K similar pairs injected as context prefix
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TOP_K = 3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class RAGTranslator:
    """
    Translation with retrieval-augmented context from translation memory.
    """

    def __init__(
        self,
        translator,         # RunyoroTranslator instance
        tm_pairs: Optional[List[Tuple[str, str]]] = None,
        tm_json_path: Optional[str] = None,
        top_k: int = TOP_K,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.translator = translator
        self.top_k = top_k
        self.embedding_model_name = embedding_model

        self._faiss_index = None
        self._embedder = None
        self._tm_pairs: List[Tuple[str, str]] = []

        if tm_json_path:
            self._load_tm_json(tm_json_path)
        elif tm_pairs:
            self._tm_pairs = tm_pairs

        if self._tm_pairs:
            self._build_faiss_index()

    def _load_tm_json(self, path: str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._tm_pairs = [(item["runyoro"], item["english"]) for item in data]
        logger.info("TM loaded from JSON: %d pairs", len(self._tm_pairs))

    def _build_faiss_index(self) -> None:
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as np

            logger.info("Building FAISS index over TM (%d pairs)...", len(self._tm_pairs))
            self._embedder = SentenceTransformer(self.embedding_model_name)

            sources = [r for r, e in self._tm_pairs]
            embeddings = self._embedder.encode(sources, convert_to_numpy=True)
            embeddings = embeddings / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
            )

            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)  # inner product = cosine after norm
            self._faiss_index.add(embeddings.astype("float32"))
            logger.info("FAISS index built: %d vectors, dim=%d", len(sources), dim)

        except ImportError:
            logger.warning(
                "faiss or sentence-transformers not installed — "
                "RAG will run without retrieval context"
            )

    def _retrieve(self, query: str) -> List[Tuple[str, str, float]]:
        """Return top-K (runyoro, english, score) from TM."""
        if self._faiss_index is None or self._embedder is None:
            return []

        import numpy as np

        q_emb = self._embedder.encode([query], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-9)

        D, I = self._faiss_index.search(q_emb.astype("float32"), self.top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= 0:
                rny, eng = self._tm_pairs[idx]
                results.append((rny, eng, float(score)))
        return results

    def _build_context_prefix(
        self, retrieved: List[Tuple[str, str, float]], direction: str
    ) -> str:
        """
        Build a few-shot context string from retrieved TM examples.
        This is prepended to the source text to guide the model.
        """
        if not retrieved:
            return ""

        lines = []
        for rny, eng, score in retrieved:
            if direction.startswith("rny"):
                lines.append(f"Runyoro: {rny} | English: {eng}")
            else:
                lines.append(f"English: {eng} | Runyoro: {rny}")

        return " || ".join(lines) + " || "

    def translate(self, text: str, direction: str = "auto") -> Dict:
        """
        Translate with TM-retrieved context prefix for better accuracy.
        """
        # Determine effective direction
        eff_direction = direction
        if direction == "auto":
            lang = self.translator._detect_language(text)
            eff_direction = "rny_to_en" if lang == "runyoro" else "en_to_rny"

        # Retrieve similar pairs
        retrieved = self._retrieve(text)

        # Build augmented input
        context_prefix = self._build_context_prefix(retrieved, eff_direction)
        augmented_input = context_prefix + text if context_prefix else text

        # Translate
        self.translator.direction = eff_direction
        result = self.translator.translate(augmented_input)
        result["rag_context"] = [
            {"runyoro": r, "english": e, "similarity": round(s, 4)}
            for r, e, s in retrieved
        ]
        result["rag_enabled"] = True

        return result
