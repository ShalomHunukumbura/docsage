import faiss
import numpy as np

class VectorStore:
	def __init__(self, dimension=384):
		self.index = faiss.IndexFlatIP(dimension)
		self.chunks = []

	def add(self, chunks, vectors):
		self.index.add(np.array(vectors).astype("float32"))
		self.chunks.extend(chunks)

	def search(self, query_vector, top_k=3):
		query = np.array([query_vector]).astype("float32")
		scores, positions = self.index.search(query, top_k)

		results = []
		for score, position in zip(scores[0], positions[0]):
			if position == -1:
				continue
			chunk = self.chunks[position]
			results.append({
				"text": chunk["text"],
				"source": chunk["source"],
				"score": float(score),
})
		return results
