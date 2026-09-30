"""Measure retrieval quality and calibrate the "I don't know" threshold.

No LLM is involved: this scores the part of the system that decides what
the LLM gets to see. A question counts as a hit at k when any of the top k
chunks comes from one of its relevant documents.

Usage:
	python eval/run_eval.py            # uses the saved index
	python eval/run_eval.py --sweep    # also rebuilds with several chunk sizes
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from docsage.config import get_settings
from docsage.embeddings import SentenceTransformerEmbedder
from docsage.index import build_index, load_index
from docsage.retriever import Retriever

HERE = Path(__file__).parent
MODES = ("dense", "keyword", "hybrid")
KS = (1, 3, 5)
THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(8)]


def load_questions() -> list[dict]:
	with open(HERE / "questions.jsonl", encoding="utf-8") as f:
		return [json.loads(line) for line in f if line.strip()]


def first_relevant_rank(retriever: Retriever, q: dict, mode: str, depth: int = 10) -> int | None:
	hits = retriever.retrieve(q["question"], top_k=depth, mode=mode).hits
	for rank, hit in enumerate(hits, start=1):
		if hit.chunk.doc_id in q["relevant"]:
			return rank
	return None


def retrieval_metrics(retriever: Retriever, questions: list[dict], mode: str) -> dict:
	ranks = [first_relevant_rank(retriever, q, mode) for q in questions]
	n = len(ranks)
	metrics = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
	metrics["mrr"] = sum(1 / r for r in ranks if r) / n
	metrics["misses"] = [q["question"] for q, r in zip(questions, ranks, strict=True) if not r or r > 5]
	return metrics


def threshold_table(retriever: Retriever, answerable: list[dict], unanswerable: list[dict]) -> list[dict]:
	def best(q):
		return retriever.retrieve(q["question"], top_k=1, mode="dense").best_similarity

	ans = [best(q) for q in answerable]
	una = [best(q) for q in unanswerable]
	rows = []
	for t in THRESHOLDS:
		answered = sum(s >= t for s in ans) / len(ans)
		refused = sum(s < t for s in una) / len(una)
		rows.append({"threshold": t, "answered": answered, "refused": refused, "balanced": (answered + refused) / 2})
	return rows


def pct(x: float) -> str:
	return f"{x * 100:.0f}%"


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--sweep", action="store_true", help="compare chunk sizes (rebuilds temporary indexes)")
	args = parser.parse_args()

	settings = get_settings()
	questions = load_questions()
	answerable = [q for q in questions if q["relevant"]]
	unanswerable = [q for q in questions if not q["relevant"]]

	embedder = SentenceTransformerEmbedder(settings.embedding_model)
	chunks, store, meta = load_index(settings.index_dir)
	retriever = Retriever(chunks, store, embedder)

	out = [
		"# Evaluation results",
		"",
		f"{len(answerable)} answerable and {len(unanswerable)} unanswerable questions "
		f"over {len(meta['documents'])} documents ({meta['chunk_count']} chunks of up to "
		f"{meta['chunk_words']} words). Embeddings: `{meta['embedding_model']}`.",
		"",
		"## Retrieval by mode",
		"",
		"| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |",
		"| --- | --- | --- | --- | --- |",
	]
	misses = {}
	for mode in MODES:
		m = retrieval_metrics(retriever, answerable, mode)
		misses[mode] = m["misses"]
		out.append(f"| {mode} | {pct(m['hit@1'])} | {pct(m['hit@3'])} | {pct(m['hit@5'])} | {m['mrr']:.3f} |")

	out += [
		"",
		"## Refusal threshold",
		"",
		"DocSage refuses without calling the LLM when the closest chunk's cosine similarity is below "
		"the threshold. *Answered* is the share of answerable questions let through; *refused* is the "
		"share of unanswerable questions stopped.",
		"",
		"| Threshold | Answered | Refused | Balanced |",
		"| --- | --- | --- | --- |",
	]
	rows = threshold_table(retriever, answerable, unanswerable)
	chosen = settings.min_similarity
	for r in rows:
		mark = " (current)" if abs(r["threshold"] - chosen) < 1e-9 else ""
		out.append(
			f"| {r['threshold']:.2f}{mark} | {pct(r['answered'])} | {pct(r['refused'])} | {pct(r['balanced'])} |"
		)

	if args.sweep:
		out += ["", "## Chunk size (hybrid)", "", "| Words per chunk | Chunks | Hit@1 | Hit@5 | MRR |",
				"| --- | --- | --- | --- | --- |"]
		for size in (100, 200, 300):
			with tempfile.TemporaryDirectory() as tmp:
				m_meta = build_index(settings.corpus_dir, Path(tmp), embedder, size, size // 5)
				c, s, _ = load_index(Path(tmp))
				m = retrieval_metrics(Retriever(c, s, embedder), answerable, "hybrid")
			out.append(f"| {size} | {m_meta['chunk_count']} | {pct(m['hit@1'])} | {pct(m['hit@5'])} | {m['mrr']:.3f} |")

	out += ["", "## Misses (hybrid, not in top 5)", ""]
	out += [f"- {q}" for q in misses["hybrid"]] or ["None."]

	report = "\n".join(out) + "\n"
	(HERE / "results.md").write_text(report, encoding="utf-8")
	print(report)
	return 0


if __name__ == "__main__":
	sys.exit(main())
