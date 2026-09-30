"""Stage 2: split sections into overlapping chunks.

Chunks are built from whole paragraphs where possible, so a list or code
block is not cut in half. A paragraph longer than the budget on its own is
split between list items or lines, and only as a last resort by words.
Chunks never cross a section boundary.
"""

import re
from dataclasses import dataclass

from docsage.loader import Document, Section

# A section shorter than this that introduces subsections ("Here are some
# examples:") is folded into its first subsection instead of standing alone.
MIN_SECTION_WORDS = 20


@dataclass
class Chunk:
	id: str
	doc_id: str
	title: str
	collection: str
	url: str
	section: str
	text: str

	def embedding_text(self) -> str:
		# Prefixing the title and section gives short chunks the context they
		# need, e.g. a "How to Prevent" list that never repeats "injection".
		heading = f"{self.title} > {self.section}" if self.section else self.title
		return f"{heading}\n\n{self.text}"


def chunk_documents(documents: list[Document], max_words: int = 200, overlap: int = 40) -> list[Chunk]:
	chunks = []
	for doc in documents:
		n = 0
		for section in _merge_intro_sections(doc.sections):
			for text in chunk_text(section.text, max_words, overlap):
				chunks.append(Chunk(
					id=f"{doc.id}#{n}",
					doc_id=doc.id,
					title=doc.title,
					collection=doc.collection,
					url=doc.url,
					section=section.heading,
					text=text,
				))
				n += 1
	return chunks


def chunk_text(text: str, max_words: int = 200, overlap: int = 40) -> list[str]:
	if overlap >= max_words:
		raise ValueError("overlap must be smaller than max_words")

	pieces: list[str] = []
	for para in _paragraphs(text):
		if _words(para) <= max_words:
			pieces.append(para)
		else:
			pieces.extend(_split_long(para, max_words, overlap))

	chunks: list[str] = []
	current: list[str] = []
	for piece in pieces:
		if current and sum(map(_words, current)) + _words(piece) > max_words:
			chunks.append("\n\n".join(current))
			current = _tail(current, overlap)
			# The carried-over tail must leave room for the new piece.
			while current and sum(map(_words, current)) + _words(piece) > max_words:
				current.pop(0)
		current.append(piece)

	if current:
		chunks.append("\n\n".join(current))
	return chunks


def _merge_intro_sections(sections: list[Section]) -> list[Section]:
	merged: list[Section] = []
	carry = ""
	for i, section in enumerate(sections):
		text = f"{carry}\n\n{section.text}" if carry else section.text
		carry = ""
		nxt = sections[i + 1] if i + 1 < len(sections) else None
		is_parent = nxt is not None and nxt.heading.startswith(f"{section.heading} > ")
		if is_parent and _words(text) < MIN_SECTION_WORDS:
			carry = text
			continue
		merged.append(Section(section.heading, text))
	return merged


def _paragraphs(text: str) -> list[str]:
	"""Split on blank lines, keeping a lead-in ("Small CLs are:") with what follows."""
	paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
	out: list[str] = []
	for para in paras:
		if out and out[-1].endswith(":"):
			out[-1] = f"{out[-1]}\n\n{para}"
		else:
			out.append(para)
	return out


def _words(text: str) -> int:
	return len(text.split())


def _split_long(text: str, size: int, overlap: int) -> list[str]:
	"""Split an oversized block on list items and lines, keeping formatting.

	Indented lines continue the item above them (wrapped bullets). A single
	item that is still too long falls back to word windows.
	"""
	items: list[str] = []
	for line in text.splitlines():
		continues = line[:1].isspace() and not line.lstrip().startswith(("-", "*"))
		if items and continues:
			items[-1] += "\n" + line
		else:
			items.append(line)

	out: list[str] = []
	current: list[str] = []
	for item in items:
		if _words(item) > size:
			if current:
				out.append("\n".join(current).strip())
				current = []
			out.extend(_split_words(item, size, overlap))
			continue
		if current and _words("\n".join(current)) + _words(item) > size:
			out.append("\n".join(current).strip())
			current = []
		current.append(item)
	if current:
		out.append("\n".join(current).strip())
	return [piece for piece in out if piece]


def _split_words(text: str, size: int, overlap: int) -> list[str]:
	words = text.split()
	out = []
	start = 0
	while True:
		out.append(" ".join(words[start:start + size]))
		if start + size >= len(words):
			return out
		start += size - overlap


def _tail(paragraphs: list[str], overlap: int) -> list[str]:
	"""Trailing whole paragraphs of the previous chunk that fit in the overlap budget."""
	tail: list[str] = []
	total = 0
	for para in reversed(paragraphs):
		total += _words(para)
		if total > overlap:
			break
		tail.insert(0, para)
	return tail
