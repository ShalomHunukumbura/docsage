def chunk_text(text, chunk_size=100, overlap=20):
	words = text.split()
	chunks = []

	start = 0
	while start < len(words):
		end = start + chunk_size
		chunk = " ".join(words[start:end])
		chunks.append(chunk)

		if end >= len(words):
			break

		start = end - overlap

	return chunks


def chunk_documents(documents, chunk_size=100, overlap=20):
	chunks = []

	for doc in documents:
		for i, piece in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
			chunks.append({
				"text": piece,
				"source": doc["source"],
				"chunk_id": i,
			})

	return chunks
