from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	"""All tunables, read from environment variables or a .env file."""

	model_config = SettingsConfigDict(env_file=".env", extra="ignore")

	corpus_dir: Path = Path("data/corpus")
	index_dir: Path = Path("index")

	embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
	chunk_words: int = 200
	chunk_overlap: int = 40

	top_k: int = 5
	# Below this cosine similarity the best match is treated as noise and the
	# question is refused without calling the LLM. Calibrated with eval/run_eval.py.
	min_similarity: float = 0.35

	# Any OpenAI-compatible endpoint works. Defaults to Groq's free tier.
	llm_base_url: str = "https://api.groq.com/openai/v1"
	llm_model: str = "openai/gpt-oss-120b"
	llm_api_key: str | None = Field(
		default=None, validation_alias=AliasChoices("LLM_API_KEY", "GROQ_API_KEY")
	)
	llm_timeout: float = 30.0


@lru_cache
def get_settings() -> Settings:
	return Settings()
