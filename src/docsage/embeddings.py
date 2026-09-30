"""Stage 3: turn text into vectors."""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
	name: str
	dimension: int

	def embed(self, texts: list[str]) -> np.ndarray:
		"""Return an (n, dimension) float32 array of L2-normalised vectors."""
		...


class SentenceTransformerEmbedder:
	def __init__(self, model_name: str):
		# Imported here so tests and tooling don't pay for loading torch.
		from sentence_transformers import SentenceTransformer

		self.name = model_name
		self.model = SentenceTransformer(model_name)
		self.dimension = self.model.get_embedding_dimension()

	def embed(self, texts: list[str]) -> np.ndarray:
		# Normalised vectors make inner product equal to cosine similarity.
		vectors = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
		return np.asarray(vectors, dtype="float32")
