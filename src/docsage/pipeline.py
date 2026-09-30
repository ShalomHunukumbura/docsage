"""Wire the pieces together from settings."""

from docsage.config import Settings, get_settings
from docsage.embeddings import SentenceTransformerEmbedder
from docsage.index import load_index
from docsage.llm import OpenAICompatibleLLM
from docsage.rag import RAG
from docsage.retriever import Retriever


def load_pipeline(settings: Settings | None = None) -> tuple[RAG, dict]:
	settings = settings or get_settings()
	chunks, store, meta = load_index(settings.index_dir)

	if meta["embedding_model"] != settings.embedding_model:
		raise ValueError(
			f"Index was built with '{meta['embedding_model']}' but EMBEDDING_MODEL is "
			f"'{settings.embedding_model}'. Rebuild it with: docsage ingest"
		)

	embedder = SentenceTransformerEmbedder(settings.embedding_model)
	llm = None
	if settings.llm_api_key:
		llm = OpenAICompatibleLLM(
			settings.llm_base_url, settings.llm_model, settings.llm_api_key, settings.llm_timeout
		)

	retriever = Retriever(chunks, store, embedder)
	return RAG(retriever, llm, settings.min_similarity, settings.top_k), meta
