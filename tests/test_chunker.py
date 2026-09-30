import pytest

from docsage.chunker import chunk_documents, chunk_text
from docsage.loader import Document, Section


def words(n, prefix="w"):
	return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_text_is_one_chunk():
	assert chunk_text("one two three", max_words=10, overlap=2) == ["one two three"]


def test_chunks_respect_word_budget():
	text = "\n\n".join(words(30, f"p{i}_") for i in range(10))
	chunks = chunk_text(text, max_words=100, overlap=30)
	assert len(chunks) > 1
	assert all(len(c.split()) <= 100 for c in chunks)


def test_paragraphs_are_not_split_when_they_fit():
	paras = [words(40, f"p{i}_") for i in range(4)]
	for chunk in chunk_text("\n\n".join(paras), max_words=100, overlap=45):
		for piece in chunk.split("\n\n"):
			assert piece in paras


def test_trailing_paragraph_overlaps_into_next_chunk():
	paras = [words(40, f"p{i}_") for i in range(4)]
	chunks = chunk_text("\n\n".join(paras), max_words=100, overlap=45)
	assert chunks[0].split("\n\n")[-1] == chunks[1].split("\n\n")[0]


def test_long_paragraph_is_split_with_word_overlap():
	chunks = chunk_text(words(250), max_words=100, overlap=20)
	assert all(len(c.split()) <= 100 for c in chunks)
	assert chunks[0].split()[-20:] == chunks[1].split()[:20]
	assert chunks[-1].split()[-1] == "w249"


def test_overlap_must_be_smaller_than_budget():
	with pytest.raises(ValueError):
		chunk_text("a b c", max_words=10, overlap=10)


def test_chunks_carry_document_and_section_metadata():
	doc = Document("c/doc", "Doc", "Coll", "https://u", "MIT", [
		Section("Intro", "hello"), Section("Intro > Deep", words(120)),
	])
	chunks = chunk_documents([doc], max_words=100, overlap=10)
	assert [c.id for c in chunks] == ["c/doc#0", "c/doc#1", "c/doc#2"]
	assert chunks[1].section == "Intro > Deep"
	assert chunks[1].embedding_text().startswith("Doc > Intro > Deep\n\n")


def test_lead_in_paragraph_stays_with_its_list():
	text = "Small CLs are:\n\n" + "\n".join(f"- item {i} " + words(20) for i in range(8)) + "\n\nAfterword."
	chunks = chunk_text(text, max_words=100, overlap=10)
	assert chunks[0].startswith("Small CLs are:\n\n- item 0")


def test_tiny_intro_section_merges_into_first_subsection():
	doc = Document("c/doc", "Doc", "Coll", "https://u", "MIT", [
		Section("Examples", "Here are some examples."),
		Section("Examples > Refactoring", "Refactor the parser."),
		Section("Other", "Standalone."),
	])
	chunks = chunk_documents([doc], max_words=100, overlap=10)
	assert [(c.section, c.text) for c in chunks] == [
		("Examples > Refactoring", "Here are some examples.\n\nRefactor the parser."),
		("Other", "Standalone."),
	]
