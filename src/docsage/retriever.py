"""Stage 5: find the chunks most relevant to a question.

Three modes, so they can be compared in eval/run_eval.py:
- dense:   cosine similarity of embeddings (meaning)
- keyword: BM25 (exact terms)
- hybrid:  both, merged with reciprocal rank fusion (default)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from docsage.bm25 import BM25
from docsage.chunker import Chunk
from docsage.embeddings import Embedder
from docsage.vector_store import VectorStore

Mode = Literal["hybrid", "dense", "keyword"]

# Standard RRF constant: dampens the gap between rank 1 and rank 2 so a
# result ranked well by both methods beats one ranked first by only one.
RRF_K = 60
# How deep each method searches before fusion.
CANDIDATES = 30


@dataclass
class Hit:
	chunk: Chunk
	score: float       # the ranking score of the chosen mode
	similarity: float  # cosine similarity to the question, comparable across modes


@dataclass
class Retrieval:
	hits: list[Hit]
	# Cosine similarity of the closest chunk in the whole index. This is the
	# signal for "the knowledge base does not cover this question".
	best_similarity: float


class Retriever:
	def __init__(self, chunks: list[Chunk], store: VectorStore, embedder: Embedder):
		self.chunks = chunks
		self.store = store
		self.embedder = embedder
		self.bm25 = BM25([c.embedding_text() for c in chunks])

	def retrieve(self, question: str, top_k: int = 5, mode: Mode = "hybrid") -> Retrieval:
		query_vector = self.embedder.embed([question])[0]
		dense = self.store.search(query_vector, max(top_k, CANDIDATES))
		best_similarity = dense[0][1] if dense else 0.0

		if mode == "dense":
			ranked = dense[:top_k]
		elif mode == "keyword":
			ranked = self.bm25.search(question, top_k)
		else:
			ranked = _reciprocal_rank_fusion([dense, self.bm25.search(question, CANDIDATES)])[:top_k]

		similarities = self._similarities(query_vector, [pos for pos, _ in ranked])
		hits = [
			Hit(chunk=self.chunks[pos], score=score, similarity=similarities[pos])
			for pos, score in ranked
		]
		return Retrieval(hits=hits, best_similarity=best_similarity)

	def _similarities(self, query_vector: np.ndarray, positions: list[int]) -> dict[int, float]:
		if not positions:
			return {}
		vectors = np.vstack([self.store.index.reconstruct(p) for p in positions])
		return dict(zip(positions, (vectors @ query_vector).tolist(), strict=True))


def _reciprocal_rank_fusion(rankings: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
	fused: dict[int, float] = {}
	for ranking in rankings:
		for rank, (pos, _) in enumerate(ranking, start=1):
			fused[pos] = fused.get(pos, 0.0) + 1.0 / (RRF_K + rank)
	return sorted(fused.items(), key=lambda item: item[1], reverse=True)
