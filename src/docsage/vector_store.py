"""Stage 4a: dense vector index (semantic search)."""

from pathlib import Path

import faiss
import numpy as np


class VectorStore:
	"""Exact inner-product search. Positions line up with the chunk list."""

	def __init__(self, dimension: int):
		self.index = faiss.IndexFlatIP(dimension)

	def __len__(self) -> int:
		return self.index.ntotal

	def add(self, vectors: np.ndarray) -> None:
		self.index.add(np.ascontiguousarray(vectors, dtype="float32"))

	def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
		query = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
		scores, positions = self.index.search(query, top_k)
		# FAISS pads with -1 when the index has fewer than top_k vectors.
		return [(int(p), float(s)) for p, s in zip(positions[0], scores[0], strict=True) if p != -1]

	def save(self, path: Path) -> None:
		faiss.write_index(self.index, str(path))

	@classmethod
	def load(cls, path: Path) -> "VectorStore":
		store = cls.__new__(cls)
		store.index = faiss.read_index(str(path))
		return store
