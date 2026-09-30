"""Build the index once, save it to disk, load it at startup.

Embedding the corpus takes a while on CPU, so the API never does it. Run
`docsage ingest` after changing the documents or chunking settings.
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from docsage.chunker import Chunk, chunk_documents
from docsage.embeddings import Embedder
from docsage.loader import load_corpus
from docsage.vector_store import VectorStore

VECTORS_FILE = "vectors.faiss"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"


class IndexNotFoundError(FileNotFoundError):
	pass


def build_index(corpus_dir: Path, index_dir: Path, embedder: Embedder,
				max_words: int, overlap: int) -> dict:
	documents = load_corpus(corpus_dir)
	chunks = chunk_documents(documents, max_words=max_words, overlap=overlap)

	store = VectorStore(embedder.dimension)
	store.add(embedder.embed([c.embedding_text() for c in chunks]))

	meta = {
		"embedding_model": embedder.name,
		"dimension": embedder.dimension,
		"chunk_words": max_words,
		"chunk_overlap": overlap,
		"documents": [
			{"id": d.id, "title": d.title, "collection": d.collection, "url": d.url, "license": d.license}
			for d in documents
		],
		"chunk_count": len(chunks),
		"built_at": datetime.now(UTC).isoformat(timespec="seconds"),
	}

	index_dir.mkdir(parents=True, exist_ok=True)
	store.save(index_dir / VECTORS_FILE)
	with open(index_dir / CHUNKS_FILE, "w", encoding="utf-8") as f:
		for chunk in chunks:
			f.write(json.dumps(asdict(chunk)) + "\n")
	(index_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
	return meta


def load_index(index_dir: Path) -> tuple[list[Chunk], VectorStore, dict]:
	if not (index_dir / META_FILE).exists():
		raise IndexNotFoundError(f"No index in '{index_dir}'. Build it first with: docsage ingest")

	meta = json.loads((index_dir / META_FILE).read_text(encoding="utf-8"))
	with open(index_dir / CHUNKS_FILE, encoding="utf-8") as f:
		chunks = [Chunk(**json.loads(line)) for line in f]
	store = VectorStore.load(index_dir / VECTORS_FILE)

	if len(store) != len(chunks):
		raise ValueError(f"Index is inconsistent: {len(store)} vectors for {len(chunks)} chunks. Rebuild it.")
	return chunks, store, meta
