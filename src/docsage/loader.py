"""Stage 1: load documents and split them into titled sections.

Sources are Markdown or reStructuredText. Headings are kept as a path
("How to Prevent > Parameterized queries") so every chunk can say where it
came from, which is what makes the citations useful.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Sections that add noise to retrieval without answering anything.
SKIPPED_SECTIONS = {"references", "copyright", "footnotes", "license"}

_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_HEADING_ANCHOR = re.compile(r"\s*\{#[^}]*\}\s*$")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_NOISE = re.compile(r"<!--.*?-->|<a id=[^>]*></a>", re.DOTALL)
_UNDERLINE = re.compile(r"^([=\-~^*+#])\1{2,}\s*$")
_RST_DIRECTIVE = re.compile(r"^\.\. (code-block::.*|_[^:]+:.*)$")


@dataclass
class Section:
	heading: str  # "Parent > Child", empty for text before the first heading
	text: str


@dataclass
class Document:
	id: str
	title: str
	collection: str
	url: str
	license: str
	sections: list[Section] = field(default_factory=list)


def load_corpus(corpus_dir: str | Path) -> list[Document]:
	"""Load every document listed in the corpus manifest."""
	corpus_dir = Path(corpus_dir)
	manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))

	documents = []
	for entry in manifest:
		path = corpus_dir / entry["path"]
		text = path.read_text(encoding="utf-8")
		documents.append(Document(
			id=entry["id"],
			title=entry["title"],
			collection=entry["collection"],
			url=entry["url"],
			license=entry["license"],
			sections=parse_sections(text, rst=path.suffix == ".rst"),
		))
	return documents


def parse_sections(text: str, rst: bool = False) -> list[Section]:
	text = _strip_rst_header(text) if rst else _FRONT_MATTER.sub("", text)
	lines = text.splitlines()

	sections: list[Section] = []
	stack: list[tuple[int, str]] = []  # (level, heading)
	body: list[str] = []
	underline_levels: dict[str, int] = {}  # rst assigns levels by order of first use
	seen_body_text = False
	in_code = False

	def flush():
		heading = " > ".join(h for _, h in stack)
		content = _clean("\n".join(body))
		leaf = stack[-1][1].lower() if stack else ""
		if content and leaf not in SKIPPED_SECTIONS:
			sections.append(Section(heading=heading, text=content))
		body.clear()

	def open_heading(level: int, heading: str):
		nonlocal seen_body_text
		# In Markdown the first heading before any body text is the document
		# title, which the manifest already has.
		if not rst and not sections and not seen_body_text and not stack:
			stack.append((0, ""))
			return
		flush()
		while stack and stack[-1][0] >= level:
			stack.pop()
		stack.append((level, heading))

	i = 0
	while i < len(lines):
		line = lines[i]
		nxt = lines[i + 1] if i + 1 < len(lines) else ""

		if line.lstrip().startswith("```"):
			in_code = not in_code
		elif not in_code:
			atx = None if rst else _ATX_HEADING.match(line)
			if atx:
				open_heading(len(atx.group(1)), _HEADING_ANCHOR.sub("", atx.group(2)))
				i += 1
				continue

			underline = _UNDERLINE.match(nxt)
			if line.strip() and underline and len(nxt.strip()) >= len(line.strip()):
				char = underline.group(1)
				if rst:
					level = underline_levels.setdefault(char, len(underline_levels) + 1)
				else:
					level = 1 if char == "=" else 2
				open_heading(level, line.strip())
				i += 2
				continue

		if line.strip():
			seen_body_text = True
		body.append(line)
		i += 1

	flush()
	# Drop the placeholder title entry from heading paths.
	for s in sections:
		s.heading = s.heading.removeprefix(" > ").strip()
	return sections


def _strip_rst_header(text: str) -> str:
	"""PEPs start with an RFC 822 style header block ("PEP: 8", "Title: ...")."""
	if re.match(r"^[A-Z][\w-]*:", text):
		_, _, text = text.partition("\n\n")
	return text


def _clean(text: str) -> str:
	text = _HTML_NOISE.sub("", text)
	text = _IMAGE.sub("", text)
	text = _LINK.sub(r"\1", text)
	lines = [
		line for line in text.splitlines()
		# Markdown tables here are stat grids (OWASP "Factors"), not prose.
		if not line.lstrip().startswith("|") and not _RST_DIRECTIVE.match(line.strip())
	]
	text = "\n".join(lines)
	return re.sub(r"\n{3,}", "\n\n", text).strip()
