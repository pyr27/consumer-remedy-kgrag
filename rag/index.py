"""코사인 유사도 기반 단순 벡터 인덱스.

Naive RAG 베이스라인이므로 재순위화(rerank)나 키워드 결합(BM25) 없이
질의 벡터와 청크 벡터의 코사인 유사도 상위 k개만 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class Hit:
    rank: int
    score: float
    chunk: Chunk


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class VectorIndex:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def build(self, chunks: list[Chunk]) -> "VectorIndex":
        if not chunks:
            raise ValueError("청크가 비어 있습니다.")
        self.chunks = chunks
        texts = [c.text for c in chunks]

        fit = getattr(self.embedder, "fit", None)
        if callable(fit):
            fit(texts)

        self._matrix = _l2_normalize(self.embedder.embed(texts))
        return self

    def search(self, query: str, top_k: int = 4) -> list[Hit]:
        if self._matrix is None:
            raise RuntimeError("build()를 먼저 호출해야 합니다.")
        q = _l2_normalize(self.embedder.embed([query]))[0]
        scores = self._matrix @ q
        order = np.argsort(-scores)[:top_k]
        return [
            Hit(rank=i + 1, score=float(scores[j]), chunk=self.chunks[j])
            for i, j in enumerate(order)
        ]
