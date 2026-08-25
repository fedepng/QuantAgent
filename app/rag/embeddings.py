from __future__ import annotations

import hashlib
import re

import numpy as np


class HashingEmbedding:
    """Deterministic multilingual token/character n-gram embedding.

    It keeps the project offline and reproducible. The vector index still uses
    cosine similarity through normalized inner products and can be replaced by
    a neural embedding provider without changing the RAG service API.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 64:
            raise ValueError("embedding dimension must be at least 64")
        self.dimension = dimension

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        words = re.findall(r"[a-z0-9_\.\-]+", normalized)
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
        ngrams = [chinese[index : index + size] for size in (1, 2, 3) for index in range(len(chinese) - size + 1)]
        return words + ngrams

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                column = value % self.dimension
                sign = 1.0 if value & 1 else -1.0
                matrix[row, column] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.where(norms == 0, 1, norms)
        return matrix

