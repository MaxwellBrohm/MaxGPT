"""A tiny, dependency-free TF-IDF retriever.

Indexes text chunks and returns the most relevant ones for a query by TF-IDF cosine
similarity. Pure numpy, no embedding model to download. Good enough for retrieving over
attachments and a modest knowledge base; a learned sentence-embedder could be swapped in
later for more semantic matching.
"""
from __future__ import annotations

import math
import re

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> list[str]:
    return _WORD.findall(s.lower())


class TfidfRetriever:
    def __init__(self):
        self.chunks: list[str] = []
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.vecs: np.ndarray | None = None

    def add(self, chunks: list[str]) -> None:
        self.chunks.extend(c for c in chunks if c.strip())
        self._build()

    def _build(self) -> None:
        docs = [_tokenize(c) for c in self.chunks]
        self.vocab = {}
        for d in docs:
            for w in d:
                self.vocab.setdefault(w, len(self.vocab))
        V, N = len(self.vocab), len(docs)
        if V == 0:
            self.vecs = None
            return
        tf = np.zeros((N, V), dtype=np.float32)
        df = np.zeros(V, dtype=np.float32)
        for i, d in enumerate(docs):
            seen = set()
            for w in d:
                j = self.vocab[w]
                tf[i, j] += 1.0
                if w not in seen:
                    df[j] += 1.0
                    seen.add(w)
        self.idf = np.log((1 + N) / (1 + df)) + 1.0
        vecs = tf * self.idf
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vecs = vecs / norms

    def query(self, q: str, k: int = 3) -> list[tuple[str, float]]:
        if self.vecs is None or not self.chunks:
            return []
        qv = np.zeros(len(self.vocab), dtype=np.float32)
        for w in _tokenize(q):
            j = self.vocab.get(w)
            if j is not None:
                qv[j] += 1.0
        qv *= self.idf
        n = np.linalg.norm(qv)
        if n == 0:
            return []
        qv /= n
        sims = self.vecs @ qv
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order if sims[i] > 0]
