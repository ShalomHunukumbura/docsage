from docsage.loader import load_corpus, parse_sections


def test_markdown_title_heading_is_not_a_section():
	sections = parse_sections("# Title\n\nIntro text.\n\n## Part\n\nBody.\n")
	assert [(s.heading, s.text) for s in sections] == [("", "Intro text."), ("Part", "Body.")]


def test_heading_path_follows_nesting():
	md = "# T\n\n## A\n\na\n\n### B\n\nb\n\n## C\n\nc\n"
	assert [s.heading for s in parse_sections(md)] == ["A", "A > B", "C"]


def test_front_matter_and_heading_anchors_are_stripped():
	md = "---\ndraft: false\n---\n\n# Title\n\n## First Line {#first-line}\n\ntext\n"
	assert parse_sections(md)[0].heading == "First Line"


def test_hash_inside_code_fence_is_not_a_heading():
	md = "# T\n\n## Real\n\n```\n# just a comment\n```\n"
	sections = parse_sections(md)
	assert [s.heading for s in sections] == ["Real"]
	assert "# just a comment" in sections[0].text


def test_setext_headings():
	md = "SemVer\n======\n\nSummary\n-------\n\nMAJOR.MINOR.PATCH\n"
	assert [(s.heading, s.text) for s in parse_sections(md)] == [("Summary", "MAJOR.MINOR.PATCH")]


def test_links_images_and_tables_are_cleaned():
	md = (
		"# T\n\n## S\n\nSee [the docs](https://x.y) ![icon](a.png){: style=\"h\"}\n\n"
		"| a | b |\n|---|---|\n| 1 | 2 |\n"
	)
	assert parse_sections(md)[0].text == "See the docs"


def test_references_section_is_skipped():
	md = "# T\n\n## Overview\n\nuseful\n\n## References\n\n- link\n"
	assert [s.heading for s in parse_sections(md)] == ["Overview"]


def test_rst_pep_header_and_directives():
	rst = (
		"PEP: 8\nTitle: Style\n\nIntroduction\n============\n\nHello.\n\n"
		"Indentation\n-----------\n\n.. code-block::\n\n    x = 1\n"
	)
	sections = parse_sections(rst, rst=True)
	assert [s.heading for s in sections] == ["Introduction", "Introduction > Indentation"]
	assert "code-block" not in sections[1].text
	assert "x = 1" in sections[1].text


def test_load_corpus_uses_manifest_metadata(corpus_dir):
	docs = {d.id: d for d in load_corpus(corpus_dir)}
	assert set(docs) == {"test/config", "test/injection", "test/zen"}
	assert docs["test/config"].url == "https://example.com/config.md"
	assert docs["test/config"].sections[0].heading == "Store config in the environment"
	assert docs["test/zen"].sections[0].heading == "The Zen of Python"
