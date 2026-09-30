from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from docsage import __version__
from docsage.llm import LLMError
from docsage.rag import RAG, Answer
from docsage.retriever import Hit, Mode

STATIC_DIR = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
	query: str = Field(min_length=1, max_length=500)
	top_k: int = Field(default=5, ge=1, le=10)
	mode: Mode = "hybrid"


class AskRequest(BaseModel):
	question: str = Field(min_length=1, max_length=500)
	top_k: int = Field(default=5, ge=1, le=10)


class ChunkOut(BaseModel):
	id: str
	title: str
	section: str
	collection: str
	url: str
	text: str
	score: float
	similarity: float


class SourceOut(ChunkOut):
	number: int
	cited: bool


class SearchResponse(BaseModel):
	hits: list[ChunkOut]
	best_similarity: float


class AskResponse(BaseModel):
	answer: str
	abstained: bool
	best_similarity: float
	sources: list[SourceOut]


def _chunk_out(hit: Hit) -> dict:
	c = hit.chunk
	return {
		"id": c.id, "title": c.title, "section": c.section, "collection": c.collection,
		"url": c.url, "text": c.text, "score": hit.score, "similarity": hit.similarity,
	}


def _answer_out(answer: Answer) -> AskResponse:
	return AskResponse(
		answer=answer.answer,
		abstained=answer.abstained,
		best_similarity=answer.best_similarity,
		sources=[SourceOut(number=s.number, cited=s.cited, **_chunk_out(s.hit)) for s in answer.sources],
	)


def create_app(rag: RAG | None = None, meta: dict | None = None) -> FastAPI:
	"""Pass `rag` to inject a pre-built pipeline (tests); otherwise it loads at startup."""

	@asynccontextmanager
	async def lifespan(app: FastAPI):
		if rag is None:
			from docsage.pipeline import load_pipeline

			app.state.rag, app.state.meta = load_pipeline()
		else:
			app.state.rag, app.state.meta = rag, meta or {}
		yield

	app = FastAPI(title="DocSage", version=__version__, lifespan=lifespan)

	@app.get("/", include_in_schema=False)
	def home():
		return FileResponse(STATIC_DIR / "index.html")

	@app.get("/health")
	def health(request: Request):
		state = request.app.state
		return {
			"status": "ok",
			"documents": len(state.meta.get("documents", [])),
			"chunks": len(state.rag.retriever.chunks),
			"llm": state.rag.llm.model if state.rag.llm else None,
			"min_similarity": state.rag.min_similarity,
		}

	@app.get("/api/documents")
	def documents(request: Request):
		return request.app.state.meta.get("documents", [])

	@app.post("/api/search", response_model=SearchResponse)
	def search(body: SearchRequest, request: Request):
		retrieval = request.app.state.rag.retriever.retrieve(body.query, body.top_k, body.mode)
		return {"hits": [_chunk_out(h) for h in retrieval.hits], "best_similarity": retrieval.best_similarity}

	@app.post("/api/ask", response_model=AskResponse)
	def ask(body: AskRequest, request: Request):
		try:
			answer = request.app.state.rag.ask(body.question, body.top_k)
		except LLMError as e:
			raise HTTPException(e.status_code, str(e)) from e
		return _answer_out(answer)

	return app


app = create_app()
