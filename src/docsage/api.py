from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from openai import APIConnectionError, NotFoundError

from src.rag import RAG

app = FastAPI(title="RAG Explorer")

# Built once at startup, reused for every request.
rag = RAG()


class Question(BaseModel):
	question: str
	top_k: int = 3


@app.get("/")
def home():
	return FileResponse("static/index.html")


@app.post("/ask")
def ask(q: Question):
	hits = rag.retriever.retrieve(q.question, top_k=q.top_k)

	context = "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)

	from src.rag import client, MODEL, PROMPT
	try:
		response = client.chat.completions.create(
			model=MODEL,
			messages=[{
				"role": "user",
				"content": PROMPT.format(context=context, question=q.question),
			}],
		)
	except APIConnectionError:
		raise HTTPException(
			503,
			f"Cannot reach the LLM at {client.base_url}. "
			"Start it with: ollama serve  (and: ollama pull " + MODEL + ")",
		)
	except NotFoundError:
		raise HTTPException(404, f"Model '{MODEL}' not found. Run: ollama pull {MODEL}")

	return {
		"answer": response.choices[0].message.content,
		"chunks": hits,
	}


@app.post("/retrieve")
def retrieve_only(q: Question):
	"""Retrieval without the LLM - fast, free, and works without Ollama."""
	return {"chunks": rag.retriever.retrieve(q.question, top_k=q.top_k)}
