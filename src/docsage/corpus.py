"""The knowledge base: which documents it contains and how to download them.

Every source is openly licensed (MIT, CC BY, CC BY-SA or public domain).
The manifest written alongside the files keeps title, URL and license for
each document, so answers can cite them and attribution stays intact.
"""

import json
import sys
import urllib.request
from pathlib import Path

GH = "https://raw.githubusercontent.com"

COLLECTIONS = {
	"twelve-factor": {
		"name": "The Twelve-Factor App",
		"license": "MIT",
		"docs": [
			(slug, title, f"{GH}/heroku/12factor/master/content/en/{slug}.md", f"https://12factor.net/{slug}")
			for slug, title in [
				("codebase", "I. Codebase"),
				("dependencies", "II. Dependencies"),
				("config", "III. Config"),
				("backing-services", "IV. Backing services"),
				("build-release-run", "V. Build, release, run"),
				("processes", "VI. Processes"),
				("port-binding", "VII. Port binding"),
				("concurrency", "VIII. Concurrency"),
				("disposability", "IX. Disposability"),
				("dev-prod-parity", "X. Dev/prod parity"),
				("logs", "XI. Logs"),
				("admin-processes", "XII. Admin processes"),
			]
		],
	},
	"code-review": {
		"name": "Google Engineering Practices: Code Review",
		"license": "CC BY 3.0",
		"docs": [
			(slug.replace("/", "-"), title,
			 f"{GH}/google/eng-practices/master/review/{slug}.md",
			 f"https://google.github.io/eng-practices/review/{slug}.html")
			for slug, title in [
				("reviewer/standard", "The Standard of Code Review"),
				("reviewer/looking-for", "What to Look For in a Code Review"),
				("reviewer/navigate", "Navigating a CL in Review"),
				("reviewer/speed", "Speed of Code Reviews"),
				("reviewer/comments", "How to Write Code Review Comments"),
				("reviewer/pushback", "Handling Pushback in Code Reviews"),
				("developer/cl-descriptions", "Writing Good CL Descriptions"),
				("developer/small-cls", "Small CLs"),
				("developer/handling-comments", "How to Handle Reviewer Comments"),
				("emergencies", "Emergencies"),
			]
		],
	},
	"owasp-top-10": {
		"name": "OWASP Top 10 (2021)",
		"license": "CC BY-SA 4.0",
		"docs": [
			(code.lower(), f"{code}: {title}",
			 f"{GH}/OWASP/Top10/master/2021/docs/en/{code}_2021-{page}.md",
			 f"https://owasp.org/Top10/{code}_2021-{page}/")
			for code, title, page in [
				("A01", "Broken Access Control", "Broken_Access_Control"),
				("A02", "Cryptographic Failures", "Cryptographic_Failures"),
				("A03", "Injection", "Injection"),
				("A04", "Insecure Design", "Insecure_Design"),
				("A05", "Security Misconfiguration", "Security_Misconfiguration"),
				("A06", "Vulnerable and Outdated Components", "Vulnerable_and_Outdated_Components"),
				("A07", "Identification and Authentication Failures", "Identification_and_Authentication_Failures"),
				("A08", "Software and Data Integrity Failures", "Software_and_Data_Integrity_Failures"),
				("A09", "Security Logging and Monitoring Failures", "Security_Logging_and_Monitoring_Failures"),
				("A10", "Server-Side Request Forgery (SSRF)", "Server-Side_Request_Forgery_(SSRF)"),
			]
		],
	},
	"versioning": {
		"name": "Versioning and Commit Conventions",
		"license": "CC BY 3.0",
		"docs": [
			("semver", "Semantic Versioning 2.0.0",
			 f"{GH}/semver/semver/master/semver.md", "https://semver.org/"),
			("conventional-commits", "Conventional Commits 1.0.0",
			 f"{GH}/conventional-commits/conventionalcommits.org/master/content/v1.0.0/index.md",
			 "https://www.conventionalcommits.org/en/v1.0.0/"),
		],
	},
	"python-peps": {
		"name": "Python Enhancement Proposals",
		"license": "Public domain",
		"docs": [
			(f"pep-{n}", title, f"{GH}/python/peps/main/peps/pep-{n}.rst", f"https://peps.python.org/pep-{n}/")
			for n, title in [
				("0008", "PEP 8: Style Guide for Python Code"),
				("0020", "PEP 20: The Zen of Python"),
				("0257", "PEP 257: Docstring Conventions"),
			]
		],
	},
}


def fetch(url: str) -> str:
	# urllib's default user agent is blocked by some hosts.
	req = urllib.request.Request(url, headers={"User-Agent": "docsage-fetch/1.0"})
	with urllib.request.urlopen(req, timeout=30) as resp:
		return resp.read().decode("utf-8")


def fetch_corpus(corpus_dir: Path) -> int:
	"""Download every document into corpus_dir. Returns the number of failures."""
	manifest = []
	failures = 0

	for collection, spec in COLLECTIONS.items():
		out_dir = corpus_dir / collection
		out_dir.mkdir(parents=True, exist_ok=True)

		for doc_id, title, raw_url, page_url in spec["docs"]:
			suffix = ".rst" if raw_url.endswith(".rst") else ".md"
			path = out_dir / f"{doc_id}{suffix}"
			try:
				path.write_text(fetch(raw_url), encoding="utf-8")
			except Exception as exc:  # keep going, report at the end
				print(f"  FAILED {collection}/{doc_id}: {exc}", file=sys.stderr)
				failures += 1
				continue

			manifest.append({
				"id": f"{collection}/{doc_id}",
				"path": str(path.relative_to(corpus_dir)),
				"title": title,
				"collection": spec["name"],
				"url": page_url,
				"license": spec["license"],
			})
			print(f"  {collection}/{doc_id}")

	(corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
	print(f"\n{len(manifest)} documents written to {corpus_dir}/")
	return failures
