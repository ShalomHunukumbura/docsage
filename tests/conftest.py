import hashlib
import json

import numpy as np
import pytest

from docsage.bm25 import tokenize
from docsage.chunker import chunk_documents
from docsage.loader import load_corpus
from docsage.rag import RAG
from docsage.retriever import Retriever
from docsage.vector_store import VectorStore


class FakeEmbedder:
	"""Bag-of-words hashed into a small vector: shared words mean higher similarity.

	Deterministic and instant, so tests never load a real model.
	"""

	name = "fake-embedder"
	dimension = 64

	def embed(self, texts):
		out = np.zeros((len(texts), self.dimension), dtype="float32")
		for row, text in enumerate(texts):
			for token in tokenize(text):
				bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension
				out[row, bucket] += 1.0
			norm = np.linalg.norm(out[row])
			if norm:
				out[row] /= norm
		return out


class FakeLLM:
	model = "fake-llm"

	def __init__(self, reply="Store config in environment variables [1]."):
		self.reply = reply
		self.calls = []

	def complete(self, system, user):
		self.calls.append((system, user))
		return self.reply


DOCS = {
	"config.md": """## III. Config
### Store config in the environment

The twelve-factor app stores config in environment variables. Env vars are easy
to change between deploys without changing any code.
""",
	"injection.md": """# A03:2021 – Injection

## How to Prevent

Use parameterized queries so user input is never concatenated into SQL.
Validate input on the server side.
""",
	"zen.rst": """PEP: 20
Title: The Zen of Python

The Zen of Python
=================

Beautiful is better than ugly. Explicit is better than implicit.
""",
}


@pytest.fixture
def corpus_dir(tmp_path):
	manifest = []
	for name, text in DOCS.items():
		(tmp_path / name).write_text(text, encoding="utf-8")
		manifest.append({
			"id": f"test/{name.split('.')[0]}", "path": name, "title": name.split(".")[0].title(),
			"collection": "Test", "url": f"https://example.com/{name}", "license": "CC0",
		})
	(tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
	return tmp_path


@pytest.fixture
def embedder():
	return FakeEmbedder()


@pytest.fixture
def retriever(corpus_dir, embedder):
	chunks = chunk_documents(load_corpus(corpus_dir), max_words=50, overlap=10)
	store = VectorStore(embedder.dimension)
	store.add(embedder.embed([c.embedding_text() for c in chunks]))
	return Retriever(chunks, store, embedder)


@pytest.fixture
def llm():
	return FakeLLM()


@pytest.fixture
def rag(retriever, llm):
	return RAG(retriever, llm, min_similarity=0.2, top_k=3)
