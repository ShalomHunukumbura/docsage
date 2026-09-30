from src.loader import load_documents
from src.chunker import chunk_documents
from src.embeddings import create_embeddings
from src.vector_store import VectorStore

class Retriever:
	def __init__(self, data_dir="data"):
		documents = load_documents(data_dir)
		chunks = chunk_documents(documents)

		texts = [c["text"] for c in chunks]
		vectors = create_embeddings(texts)

		self.store = VectorStore()
		self.store.add(chunks, vectors)

	def retrieve(self, question, top_k=3):
		question_vector =  create_embeddings([question])[0]
		return self.store.search(question_vector, top_k=top_k)


