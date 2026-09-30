from pathlib import Path

def load_documents(data_dir: str):
	documents = []

	for path in Path(data_dir).glob("*.txt"):
		text = path.read_text(encoding="utf-8")
		
		documents.append({
			"text":text,
			"source": path.name,
})
	return documents
