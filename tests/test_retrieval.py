import numpy as np

from docsage.bm25 import BM25, tokenize
from docsage.index import build_index, load_index
from docsage.retriever import _reciprocal_rank_fusion
from docsage.vector_store import VectorStore


def test_tokenize_keeps_technical_terms():
	assert tokenize("What is CWE-89 and SSRF in v2.0.0?") == ["cwe-89", "ssrf", "v2.0.0"]


def test_bm25_ranks_exact_term_first():
	bm25 = BM25(["config in env vars", "sql injection prevention", "sql and config"])
	assert bm25.search("injection", 3)[0][0] == 1


def test_bm25_unknown_terms_return_nothing():
	assert BM25(["alpha beta"]).search("gamma", 5) == []


def test_rrf_rewards_agreement():
	dense = [(1, 0.9), (2, 0.8), (5, 0.7)]
	keyword = [(3, 12.0), (2, 11.0), (4, 5.0)]
	fused = [pos for pos, _ in _reciprocal_rank_fusion([dense, keyword])]
	assert fused[0] == 2  # second in both beats first in only one
	assert set(fused) == {1, 2, 3, 4, 5}


def test_vector_store_skips_padding(embedder):
	store = VectorStore(embedder.dimension)
	store.add(embedder.embed(["only one vector"]))
	assert len(store.search(embedder.embed(["one"])[0], top_k=5)) == 1


def test_index_round_trip(corpus_dir, tmp_path, embedder):
	index_dir = tmp_path / "index"
	meta = build_index(corpus_dir, index_dir, embedder, max_words=50, overlap=10)
	chunks, store, loaded_meta = load_index(index_dir)

	assert loaded_meta == meta
	assert len(chunks) == len(store) == meta["chunk_count"]
	query = embedder.embed(["environment variables"])[0]
	assert chunks[store.search(query, 1)[0][0]].doc_id == "test/config"


def test_retrieve_modes_find_the_right_document(retriever):
	for mode in ("hybrid", "dense", "keyword"):
		result = retriever.retrieve("parameterized queries prevent sql injection", top_k=2, mode=mode)
		assert result.hits[0].chunk.doc_id == "test/injection", mode


def test_similarity_is_cosine_regardless_of_mode(retriever, embedder):
	question = "environment variables config"
	result = retriever.retrieve(question, top_k=1, mode="keyword")
	hit = result.hits[0]
	expected = float(embedder.embed([hit.chunk.embedding_text()])[0] @ embedder.embed([question])[0])
	assert np.isclose(hit.similarity, expected, atol=1e-5)
	assert result.best_similarity >= hit.similarity - 1e-6
