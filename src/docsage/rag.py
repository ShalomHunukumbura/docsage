"""Stage 6b: retrieve, decide whether to answer, then generate with citations."""

import re
from dataclasses import dataclass

from docsage.llm import LLM, LLMError
from docsage.retriever import Hit, Mode, Retriever

NO_ANSWER = "I don't know based on the knowledge base."
# Models sometimes paraphrase the refusal; the opening words are stable.
NO_ANSWER_PREFIX = "I don't know"

SYSTEM_PROMPT = f"""You answer software engineering questions using only the numbered sources you are given.

Rules:
- Use only information stated in the sources. Do not add outside knowledge.
- Cite every claim with its source number in square brackets, e.g. [1] or [2][3].
- If the sources do not answer the question, reply with exactly: {NO_ANSWER}
- Be concise: a short paragraph or a few bullet points."""

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
	number: int
	hit: Hit
	cited: bool


@dataclass
class Answer:
	answer: str
	sources: list[Source]
	# True when DocSage declined to answer, either before calling the LLM
	# (retrieval found nothing close enough) or because the LLM said so.
	abstained: bool
	best_similarity: float


class RAG:
	def __init__(self, retriever: Retriever, llm: LLM | None, min_similarity: float, top_k: int = 5):
		self.retriever = retriever
		self.llm = llm
		self.min_similarity = min_similarity
		self.top_k = top_k

	def ask(self, question: str, top_k: int | None = None, mode: Mode = "hybrid") -> Answer:
		retrieval = self.retriever.retrieve(question, top_k or self.top_k, mode)

		# Retrieval always returns something, even for "best pizza recipe".
		# Refusing here is cheaper and more reliable than trusting the prompt.
		if retrieval.best_similarity < self.min_similarity or not retrieval.hits:
			sources = [Source(i, h, cited=False) for i, h in enumerate(retrieval.hits, start=1)]
			return Answer(NO_ANSWER, sources, abstained=True, best_similarity=retrieval.best_similarity)

		if self.llm is None:
			raise LLMError("No LLM is configured. Set LLM_API_KEY to enable answers.", 503)

		text = self.llm.complete(SYSTEM_PROMPT, build_prompt(question, retrieval.hits)).strip()

		cited = {int(n) for n in _CITATION.findall(text)}
		sources = [Source(i, h, cited=i in cited) for i, h in enumerate(retrieval.hits, start=1)]
		return Answer(
			answer=text,
			sources=sources,
			abstained=text.startswith(NO_ANSWER_PREFIX),
			best_similarity=retrieval.best_similarity,
		)


def build_prompt(question: str, hits: list[Hit]) -> str:
	blocks = []
	for i, hit in enumerate(hits, start=1):
		c = hit.chunk
		where = f"{c.title} > {c.section}" if c.section else c.title
		blocks.append(f"[{i}] {where}\n{c.text}")
	return "Sources:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQuestion: {question}"
