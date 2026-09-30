from src.retriever import Retriever

r =  Retriever()

for q in ["How do I build a web api?", "What is the best database to go with FastAPI?"]:
	print(f"\nQ: {q}")
	for hit in r.retrieve(q):
		print(f"  {hit['score']:.3f}  {hit['source']}")
