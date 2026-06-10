"""Retrieval index over chunks — BM25 by default, optional dense/hybrid."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Sequence

from chunk import Chunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+", text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        corpus = [_tokenize(c.content + " " + c.file + " " + c.symbol) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 20) -> list[Chunk]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0]


class DenseIndex:
    """Dense retrieval using sentence-transformers embeddings.

    Requires optional dependencies:
    - sentence-transformers
    - numpy
    """

    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "Dense retrieval requires sentence-transformers and numpy. "
                "Install them with: pip install sentence-transformers numpy"
            ) from e

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        texts = [c.content[:512] for c in chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        self._np = np

    def search(self, query: str, k: int = 20) -> list[Chunk]:
        q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        # cosine similarity
        norms = self._np.linalg.norm(self.embeddings, axis=1) * self._np.linalg.norm(q_emb)
        norms = self._np.where(norms == 0, 1e-9, norms)
        scores = self.embeddings.dot(q_emb) / norms
        ranked = self._np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in ranked]


class HybridIndex:
    """Fuse BM25 + dense using reciprocal rank fusion."""

    def __init__(self, chunks: list[Chunk], dense_model: str = "all-MiniLM-L6-v2"):
        self.bm25 = BM25Index(chunks)
        self.dense = DenseIndex(chunks, dense_model)
        self.chunks = chunks

    def search(self, query: str, k: int = 20) -> list[Chunk]:
        K = 60  # RRF constant
        bm25_results = self.bm25.search(query, k=k * 2)
        dense_results = self.dense.search(query, k=k * 2)

        scores: dict[str, float] = {}
        for rank, chunk in enumerate(bm25_results):
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (K + rank + 1)
        for rank, chunk in enumerate(dense_results):
            scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (K + rank + 1)

        id_to_chunk = {c.id: c for c in self.chunks}
        ranked = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        return [id_to_chunk[cid] for cid in ranked[:k] if cid in id_to_chunk]


def build_index(
    chunks: list[Chunk],
    mode: str = "bm25",
    dense_model: str = "all-MiniLM-L6-v2",
) -> BM25Index | DenseIndex | HybridIndex:
    if mode == "dense":
        return DenseIndex(chunks, dense_model)
    if mode == "hybrid":
        return HybridIndex(chunks, dense_model)
    return BM25Index(chunks)
