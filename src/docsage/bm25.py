"""Stage 4b: keyword index (BM25).

Dense embeddings are good at meaning but weak at exact terms such as
"SSRF", "CWE-89" or "E501". BM25 covers that gap; the retriever fuses both.
"""

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

STOPWORDS = frozenset("""
a an and are as at be but by can do does for from has have how i if in into is it its
of on or should so that the their then there these this to was what when where which
who why will with you your
""".split())


def tokenize(text: str) -> list[str]:
	return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
	def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
		self.k1 = k1
		self.b = b
		self.term_freqs = [Counter(tokenize(doc)) for doc in documents]
		self.lengths = [sum(tf.values()) for tf in self.term_freqs]
		self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

		doc_freq: Counter[str] = Counter()
		for tf in self.term_freqs:
			doc_freq.update(tf.keys())
		n = len(documents)
		self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

	def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
		terms = [t for t in tokenize(query) if t in self.idf]
		if not terms:
			return []

		scores = []
		for pos, tf in enumerate(self.term_freqs):
			norm = self.k1 * (1 - self.b + self.b * self.lengths[pos] / self.avg_length)
			score = sum(
				self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
				for t in terms if t in tf
			)
			if score > 0:
				scores.append((pos, score))

		scores.sort(key=lambda item: item[1], reverse=True)
		return scores[:top_k]
