"""Command line entry point: docsage {fetch,ingest,search,ask,serve}."""

import argparse
import sys
import textwrap

from docsage.config import get_settings


def cmd_fetch(args) -> int:
	from docsage.corpus import fetch_corpus

	return 1 if fetch_corpus(get_settings().corpus_dir) else 0


def cmd_ingest(args) -> int:
	from docsage.embeddings import SentenceTransformerEmbedder
	from docsage.index import build_index

	s = get_settings()
	print(f"Embedding with {s.embedding_model} ...")
	meta = build_index(
		s.corpus_dir, s.index_dir, SentenceTransformerEmbedder(s.embedding_model),
		s.chunk_words, s.chunk_overlap,
	)
	print(f"Indexed {len(meta['documents'])} documents as {meta['chunk_count']} chunks into {s.index_dir}/")
	return 0


def cmd_search(args) -> int:
	from docsage.pipeline import load_pipeline

	rag, _ = load_pipeline()
	result = rag.retriever.retrieve(args.query, args.top_k, args.mode)
	print(f"best similarity: {result.best_similarity:.3f}\n")
	for i, hit in enumerate(result.hits, start=1):
		c = hit.chunk
		print(f"{i}. [{hit.similarity:.3f}] {c.title} > {c.section or '(intro)'}")
		print(textwrap.indent(textwrap.shorten(c.text, 220), "   "))
	return 0


def cmd_ask(args) -> int:
	from docsage.llm import LLMError
	from docsage.pipeline import load_pipeline

	rag, _ = load_pipeline()
	try:
		answer = rag.ask(args.question, args.top_k)
	except LLMError as e:
		print(f"error: {e}", file=sys.stderr)
		return 1

	print(answer.answer + "\n")
	for s in answer.sources:
		mark = "*" if s.cited else " "
		c = s.hit.chunk
		print(f" {mark}[{s.number}] {c.title} > {c.section or '(intro)'}  {c.url}")
	return 0


def cmd_serve(args) -> int:
	import uvicorn

	uvicorn.run("docsage.api:app", host=args.host, port=args.port, reload=args.reload)
	return 0


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(prog="docsage", description=__doc__)
	sub = parser.add_subparsers(dest="command", required=True)

	sub.add_parser("fetch", help="download the knowledge base documents").set_defaults(func=cmd_fetch)
	sub.add_parser("ingest", help="chunk and embed the corpus, save the index").set_defaults(func=cmd_ingest)

	p = sub.add_parser("search", help="retrieval only, no LLM")
	p.add_argument("query")
	p.add_argument("-k", "--top-k", type=int, default=5)
	p.add_argument("-m", "--mode", choices=["hybrid", "dense", "keyword"], default="hybrid")
	p.set_defaults(func=cmd_search)

	p = sub.add_parser("ask", help="answer a question with citations")
	p.add_argument("question")
	p.add_argument("-k", "--top-k", type=int, default=None)
	p.set_defaults(func=cmd_ask)

	p = sub.add_parser("serve", help="run the web UI and API")
	p.add_argument("--host", default="127.0.0.1")
	p.add_argument("--port", type=int, default=8000)
	p.add_argument("--reload", action="store_true")
	p.set_defaults(func=cmd_serve)

	args = parser.parse_args(argv)
	return args.func(args)


if __name__ == "__main__":
	sys.exit(main())
