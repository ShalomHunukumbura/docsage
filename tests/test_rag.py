import pytest

from docsage.llm import LLMError
from docsage.rag import NO_ANSWER, RAG, build_prompt


def test_answers_with_numbered_sources(rag, llm):
	answer = rag.ask("How should config be stored?")

	assert not answer.abstained
	assert answer.answer == llm.reply
	_, prompt = llm.calls[0]
	assert "[1] " in prompt and "Question: How should config be stored?" in prompt
	assert [s.number for s in answer.sources] == list(range(1, len(answer.sources) + 1))


def test_marks_only_cited_sources(rag, llm):
	llm.reply = "Use env vars [1]."
	answer = rag.ask("How should config be stored?")
	assert [s.cited for s in answer.sources] == [True] + [False] * (len(answer.sources) - 1)


def test_abstains_without_calling_llm_when_nothing_is_relevant(retriever, llm):
	rag = RAG(retriever, llm, min_similarity=0.99)
	answer = rag.ask("best pizza recipe")

	assert answer.abstained
	assert answer.answer == NO_ANSWER
	assert llm.calls == []


def test_llm_refusal_is_reported_as_abstained(rag, llm):
	llm.reply = "I don't know based on the knowledge base."
	assert rag.ask("How should config be stored?").abstained


def test_missing_llm_is_a_clear_error(retriever):
	with pytest.raises(LLMError) as exc:
		RAG(retriever, llm=None, min_similarity=0.0).ask("config")
	assert exc.value.status_code == 503


def test_prompt_includes_title_and_section(retriever):
	hits = retriever.retrieve("environment variables", top_k=1).hits
	assert build_prompt("q", hits).startswith("Sources:\n\n[1] Config > Store config in the environment\n")
