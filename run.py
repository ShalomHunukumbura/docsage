from src.rag import RAG

rag = RAG()

questions = [
	"How do I build a web API in Python?",
	"What database can store embeddings?",
	"What is the best pizza recipe?",      # nothing in our docs - should say "I don't know"
]

for q in questions:
	result = rag.ask(q)
	print(f"\nQ: {q}")
	print(f"A: {result['answer']}")
	print(f"   sources: {result['sources']}")
