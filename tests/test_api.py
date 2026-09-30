import pytest
from fastapi.testclient import TestClient

from docsage.api import create_app
from docsage.llm import LLMError
from docsage.rag import RAG


@pytest.fixture
def client(rag):
	meta = {"documents": [{"id": "test/config", "title": "Config"}]}
	with TestClient(create_app(rag, meta)) as c:
		yield c


def test_health(client):
	body = client.get("/health").json()
	assert body["status"] == "ok"
	assert body["documents"] == 1
	assert body["llm"] == "fake-llm"


def test_ui_is_served(client):
	res = client.get("/")
	assert res.status_code == 200
	assert "DocSage" in res.text


def test_search(client):
	res = client.post("/api/search", json={"query": "sql injection", "top_k": 2, "mode": "keyword"})
	assert res.status_code == 200
	hits = res.json()["hits"]
	assert hits[0]["title"] == "Injection"
	assert {"section", "url", "text", "similarity"} <= hits[0].keys()


def test_ask_returns_answer_and_sources(client):
	body = client.post("/api/ask", json={"question": "How should config be stored?"}).json()
	assert body["answer"].endswith("[1].")
	assert body["sources"][0]["cited"] is True
	assert body["abstained"] is False


@pytest.mark.parametrize("payload", [
	{"question": ""},
	{"question": "x" * 501},
	{"question": "ok", "top_k": 0},
	{"question": "ok", "top_k": 11},
])
def test_ask_validates_input(client, payload):
	assert client.post("/api/ask", json=payload).status_code == 422


def test_search_rejects_unknown_mode(client):
	assert client.post("/api/search", json={"query": "x", "mode": "magic"}).status_code == 422


def test_llm_errors_become_http_errors(retriever):
	class BrokenLLM:
		model = "broken"

		def complete(self, system, user):
			raise LLMError("provider down", 503)

	with TestClient(create_app(RAG(retriever, BrokenLLM(), min_similarity=0.0))) as c:
		res = c.post("/api/ask", json={"question": "config"})
	assert res.status_code == 503
	assert res.json()["detail"] == "provider down"
