from src.loader import load_documents
from src.chunker import chunk_documents
from src.embeddings import create_embeddings
from src.vector_store import VectorStore

docs = load_documents("data")
chunks = chunk_documents(docs)
vectors =  create_embeddings([c["text"] for c in chunks])

store = VectorStore()
store.add(chunks, vectors)
print("stored:", store.index.ntotal, "vectors")

query = "What database should I use?"
query_vector = create_embeddings([query])[0]

for r in store.search(query_vector, top_k=3):
	print(f"{r['score']:.3f} {r['source']}")
