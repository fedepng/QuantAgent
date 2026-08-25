from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.backend = "numpy"
        self._matrix = np.empty((0, dimension), dtype="float32")
        try:
            import faiss  # type: ignore

            self._faiss = faiss.IndexFlatIP(dimension)
            self.backend = "faiss"
        except ImportError:
            self._faiss = None

    def rebuild(self, matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError("invalid embedding matrix shape")
        self._matrix = matrix
        if self._faiss is not None:
            self._faiss.reset()
            if len(matrix):
                self._faiss.add(matrix)

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if len(self._matrix) == 0:
            return []
        top_k = min(max(1, top_k), len(self._matrix))
        query = np.asarray(query, dtype="float32").reshape(1, -1)
        if self._faiss is not None:
            scores, indexes = self._faiss.search(query, top_k)
            return [(int(index), float(score)) for index, score in zip(indexes[0], scores[0]) if index >= 0]
        scores = self._matrix @ query[0]
        indexes = np.argsort(-scores)[:top_k]
        return [(int(index), float(scores[index])) for index in indexes]

