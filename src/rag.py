import os
from dotenv import load_dotenv
from openai import OpenAI
from src.retriever import Retriever

load_dotenv()

# Ollama runs a local server that speaks the OpenAI API format,
# so the same client library works - it just points at localhost.
client = OpenAI(
	base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
	api_key=os.getenv("OPENAI_API_KEY", "ollama"),
)

MODEL = os.getenv("LLM_MODEL", "llama3.2")

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say "I don't know."
Cite the source filename for each fact you use.

Context:
{context}

Question: {question}"""


class RAG:
	def __init__(self, data_dir="data"):
		self.retriever = Retriever(data_dir)

	def ask(self, question, top_k=3):
		hits = self.retriever.retrieve(question, top_k=top_k)

		context = "\n\n".join(
			f"[{h['source']}]\n{h['text']}" for h in hits
		)

		response = client.chat.completions.create(
			model=MODEL,
			messages=[{
				"role": "user",
				"content": PROMPT.format(context=context, question=question),
			}],
		)

		return {
			"answer": response.choices[0].message.content,
			"sources": [h["source"] for h in hits],
		}
