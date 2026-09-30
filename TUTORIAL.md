# Building a RAG System From Scratch

A hands-on tutorial. You write the code, this file explains what and why.

---

## What is RAG, in plain language?

Imagine you hire a brilliant assistant who has read most of the internet — but
stopped reading two years ago, and has never seen your company's documents.

You ask: *"What's our refund policy?"*

They don't know. They might **guess** and sound confident while being wrong.
(In AI, this confident guessing is called **hallucination**.)

Now imagine instead you do this:

1. You keep a filing cabinet of your company documents.
2. When someone asks a question, you **run to the cabinet** and pull out the
   3 most relevant pages.
3. You hand the assistant those pages and say: *"Answer using only these."*

That's RAG. **R**etrieval **A**ugmented **G**eneration.

- **Retrieval** = running to the filing cabinet
- **Augmented** = adding those pages to the question
- **Generation** = the assistant writing the answer

The assistant's intelligence is unchanged. What changed is that it now has the
right pages open in front of it. That's the entire idea.

### Why not just paste all the documents into the question?

Two reasons.

**Cost and size.** LLMs have a limit on how much text you can send at once
(the *context window*). A company wiki won't fit. And you pay per word sent,
so sending 500 pages to answer one question is wasteful.

**Accuracy.** Counterintuitively, more text often makes answers *worse*. Bury
the one relevant paragraph inside 500 pages of noise and the model loses it.
Handing over 3 relevant pages beats handing over 500 mixed ones.

---

## The 6 stages

```
1. LOAD      read your files into memory              ✅ done - loader.py
2. CHUNK     cut them into small pieces               ✅ done - chunker.py
3. EMBED     turn each piece into a list of numbers   ✅ done - embeddings.py
4. INDEX     store those numbers for fast searching   ← you are here
5. RETRIEVE  find the pieces closest to a question    ← next
6. GENERATE  send question + pieces to an LLM         ← last
```

Stages 1–3 are **preparation**. You run them once, when documents change.
Stages 5–6 are **query time**. They run every time someone asks a question.

Think of it as a library: stages 1–3 are cataloguing the books onto shelves.
Stages 5–6 are a visitor walking in and asking for a book.

---

## Recap: what you already built

### Stage 1 — `loader.py`

Reads every `.txt` file in `data/` into a list of dictionaries:

```python
{"text": "PostgreSQL is an open-source...", "source": "postgres.txt"}
```

**The important part is `source`.** You kept the filename. At the very end,
your app will say *"according to postgres.txt"*. If you had thrown the filename
away here, you could never get it back — you'd have text with no idea where it
came from. Like photocopying pages from books and forgetting which book each
page came from.

This idea has a name: **provenance** (where something came from). Every stage
from here on must carry it forward.

### Stage 2 — `chunker.py`

Cuts each document into overlapping 100-word pieces.

**Why cut at all?** Three reasons:

1. **Precision.** A vector represents the *average* meaning of its text. Embed a
   whole 50-page manual as one vector and you get the average of every topic in
   it — which points at nothing in particular and matches nothing well. Like
   describing an entire supermarket with one word. "Food"? Useless for finding
   the milk.

2. **It has to fit in the prompt.** At stage 6 you paste retrieved text into a
   question. You can't paste 50 pages.

3. **The embedding model has a hard limit.** `all-MiniLM-L6-v2` reads only the
   first ~200 words of anything you give it. **The rest is silently thrown
   away** — no error, no warning. Feed it a whole document and most of that
   document is never embedded at all. This one surprises people.

**Why overlap?** Say you cut strictly every 100 words and a sentence lands
across the cut:

```
chunk 0: "...PostgreSQL can be extended with extensions such"
chunk 1: "as pgvector for storing and searching embeddings."
```

Now *neither* chunk means anything. The first trails off; the second starts
mid-thought. A question about pgvector matches neither well.

Overlap fixes it by making consecutive chunks share their edges:

```
chunk 0: words   0 → 99
chunk 1: words  80 → 179     starts 20 words BEFORE chunk 0 ended
chunk 2: words 160 → 249
```

Words 80–99 appear in **both** chunk 0 and chunk 1. So a sentence sitting on
the boundary stays whole in at least one of them.

It's like tearing a long strip of paper but making each piece slightly longer
so the torn words are readable on one side or the other.

**A rule to remember:** `overlap` must be smaller than `chunk_size`. If they're
equal, each new window starts exactly where the last one started and the loop
runs forever.

### Stage 3 — `embeddings.py`

This is the heart of RAG, so it's worth understanding properly.

**An embedding turns text into a list of numbers that represents its meaning.**

Picture a map of the world, but instead of cities it holds *ideas*. Every piece
of text gets a coordinate. Text about similar things lands in the same
neighbourhood:

```
        "dog"  "puppy"
           •  •
                          "database"  "PostgreSQL"
                              •    •

  "car"
    •           "web framework"  "FastAPI"
                      •       •
```

A real map needs 2 numbers (latitude, longitude). This map of meaning needs
**384 numbers**. That's what "384 dimensions" means — you can't picture it, but
the maths works exactly the same as on a 2-D map. Close together = similar
meaning.

Here's the magic that makes RAG work. These two sentences:

- *"How do I build a web API?"*
- *"FastAPI is a framework for building APIs using Python."*

share almost **no words in common**. Old-fashioned keyword search finds nothing.
But their coordinates are close, because they're *about* the same thing. That's
why you can ask questions in your own words instead of guessing exact keywords.

**Why `normalize_embeddings=True` matters.** Normalizing sets every vector's
length to exactly 1.0 — it pushes every point onto the surface of a sphere,
keeping its direction but standardizing its distance from centre.

Why bother? Because it makes comparing two vectors extremely cheap. Once every
vector has length 1, **multiplying two vectors together (the "dot product")
gives you exactly their similarity score**, from -1 to 1.

Remember this — stage 4 depends on it completely. Actually depends on it.

You already saw it work:

```
similarity between "Python is a programming language"
                and "FastAPI builds web APIs"        = 0.09
```

0.09 is near zero — unrelated topics. Two sentences about the same thing would
score 0.6–0.9. Identical text scores 1.0.

**Why the function takes a list, not a single string.** Embedding models process
many texts at once far faster than one at a time — the same way an oven bakes
12 cookies   in the time it takes to bake 1. Always embed in batches.

---

# STAGE 4 — The Index

## What you're building and why

You have chunks. You'll turn each into a vector. Now: **given a question's
vector, how do you find the closest chunk vectors?**

With 3 chunks, you'd just compare against all 3 and pick the best. That's
called a **brute-force** or **flat** search. It's simple and it is always
exactly right.

With 10 million chunks, comparing against all 10 million on every question gets
too slow. Specialised libraries then use clever shortcuts that check only a
promising fraction — much faster, occasionally missing the true best match.

**FAISS** (from Meta) is the library that does both. You'll use the exact kind,
because your dataset is tiny and exactness is free.

Think of it as the difference between:
- **Flat:** reading every book title in the library. Slow, but never misses.
- **Approximate:** going straight to the shelf you *think* is right. Fast,
  occasionally wrong shelf.

## The code — create `src/vector_store.py`

```python
import faiss
import numpy as np


class VectorStore:
	def __init__(self, dimension=384):
		self.index = faiss.IndexFlatIP(dimension)
		self.chunks = []

	def add(self, chunks, vectors):
		self.index.add(np.array(vectors).astype("float32"))
		self.chunks.extend(chunks)

	def search(self, query_vector, top_k=3):
		query = np.array([query_vector]).astype("float32")
		scores, positions = self.index.search(query, top_k)

		results = []
		for score, position in zip(scores[0], positions[0]):
			if position == -1:
				continue
			chunk = self.chunks[position]
			results.append({
				"text": chunk["text"],
				"source": chunk["source"],
				"score": float(score),
			})

		return results
```

## Line-by-line explanation

### `faiss.IndexFlatIP(dimension)`

Two things are packed into that name:

- **`Flat`** = brute force. Compare against everything. Always exactly right.
- **`IP`** = **I**nner **P**roduct — the "multiply two vectors together" operation.

**This is where stage 3 comes back.** Inner product only equals *similarity*
when vectors have length 1. You normalized in `embeddings.py`, so it does.

If you ever remove `normalize_embeddings=True`, this index keeps running and
keeps returning results — **they'll just quietly be in the wrong order**. No
error, no crash, no warning. It's the most common bug in hand-built RAG systems,
because nothing tells you it happened. The two settings are a matched pair.

`dimension=384` must match your model's output. `all-MiniLM-L6-v2` produces 384
numbers, so the index expects 384. Mismatch here crashes immediately (which is
the good kind of bug — loud).

### `self.chunks = []`

**This list is essential, and here's the part that trips everyone up.**

FAISS stores *only numbers*. It has no idea your vectors came from text. When
you search, it does not return text — it returns **positions**, like
`[2, 0, 1]`, meaning *"the 3rd, 1st, and 2nd vectors you gave me."*

So you must keep your own list in **exactly the same order** as what you fed
FAISS. Position 2 in FAISS's world must be position 2 in `self.chunks`.

Like a coat check: the desk gives you ticket #47. The ticket isn't the coat.
You need the rack, in order, to turn #47 back into your coat.

This is why `add()` does both operations together — adding vectors to FAISS
and chunks to the list, in one method. Separate them and they will eventually
drift out of sync, and your app will confidently cite the wrong document.

### `.astype("float32")`

FAISS is written in C++ and requires 32-bit floats. NumPy often defaults to
64-bit (`float64`). Passing the wrong type is an error — this line makes it
explicit and safe.

`np.array([query_vector])` in `search` has **deliberate square brackets**:
FAISS's search expects a *list of queries* so you can search many at once. You
have one query, so you wrap it in a list of length 1. That's also why results
come back as `scores[0]` and `positions[0]` — you're reading the answer for
query number 0.

### `if position == -1: continue`

If you ask for `top_k=5` but only 3 vectors exist, FAISS pads the leftover
slots with `-1` meaning *"nothing here."*

Without this check, `self.chunks[-1]` would run — and in Python, index `-1`
means **the last item in the list**. You'd silently return a real-looking but
completely wrong chunk. Python's helpfulness works against you here, so guard
against it.

### Returning the score

Passing the score back lets you judge answer quality later. If the best match
scores 0.12, your documents probably don't contain the answer at all — and
it's better to say *"I don't know"* than to answer from a bad match.

## Test it

Create `test_stage4.py` in the project root:

```python
from src.loader import load_documents
from src.chunker import chunk_documents
from src.embeddings import create_embeddings
from src.vector_store import VectorStore

docs = load_documents("data")
chunks = chunk_documents(docs)
vectors = create_embeddings([c["text"] for c in chunks])

store = VectorStore()
store.add(chunks, vectors)
print("stored:", store.index.ntotal, "vectors")

query = "How do I build a web API?"
query_vector = create_embeddings([query])[0]

for r in store.search(query_vector, top_k=3):
	print(f"{r['score']:.3f}  {r['source']}")
```

Run it:

```bash
.venv/bin/python test_stage4.py
```

**What you should see:** `fastapi.txt` ranked first with the highest score.

Look closely at that result. Your question contained the words *"build"* and
*"web API"*. The FastAPI document says *"web framework for building APIs"*.
Different wording, same meaning — and the vectors found it. Keyword search
would have struggled; this didn't.

Try a few more and watch the scores move:

```python
"What database should I use?"        →  postgres.txt should win
"How does Python handle indentation?" →  python.txt should win
"What is the best pizza recipe?"      →  low scores everywhere - nothing matches
```

**That last one is the important experiment.** RAG *always* returns its top 3,
even when your documents contain nothing relevant. The scores are your only
warning sign. Run it and see how low they go — that number is what you'd use to
decide when to answer *"I don't know."*

---

# STAGE 5 — Retrieval

## What you're building

Stage 4 gave you the search machinery. Stage 5 wraps the whole pipeline into
one clean function: **question in, relevant chunks out.**

The key insight: **the question must be embedded by the same model as the
chunks.** Two different models produce two different, incompatible maps of
meaning — coordinates from one are meaningless on the other. Like giving GPS
coordinates to someone reading a map of a different planet.

## The code — create `

```python
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
		question_vector = create_embeddings([question])[0]
		return self.store.search(question_vector, top_k=top_k)
```

## Explanation

**`__init__` runs the whole preparation pipeline once.** Load → chunk → embed →
index. This is the expensive part (embedding takes real time), so you do it once
at startup, not per question. Build the library catalogue once; serve many
visitors.

**`retrieve` is the cheap part.** Embed one short question, compare, return.
Milliseconds.

**`create_embeddings([question])[0]`** — read this carefully, it's easy to get
wrong. The `[question]` wraps your string in a list, because the function takes
batches. The `[0]` unwraps the single result back out. In → list of 1.
Out → take item 0.

**`top_k` is a real trade-off, not a detail:**

- **Too low (1):** if the best chunk doesn't contain the answer, you fail
  entirely.
- **Too high (20):** you bury the good chunk in noise, cost more, and answers
  get *worse*. More context is not better context.

3–5 is the usual sweet spot. Worth experimenting with once it all works.

## Test it

```python
from src.retriever import Retriever

r = Retriever()

for q in ["How do I build a web API?", "What database supports vectors?"]:
	print(f"\nQ: {q}")
	for hit in r.retrieve(q):
		print(f"  {hit['score']:.3f}  {hit['source']}")
```

---

# STAGE 6 — Generation

## What you're building

The final step. You have relevant chunks. Now hand them to an LLM along with
the question and ask it to answer **using only those chunks**.

## The code — create `src/rag.py`

```python
import os
from anthropic import Anthropic
from src.retriever import Retriever

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

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

		message = client.messages.create(
			model="claude-sonnet-4-5",
			max_tokens=1024,
			messages=[{
				"role": "user",
				"content": PROMPT.format(context=context, question=question),
			}],
		)

		return {
			"answer": message.content[0].text,
			"sources": [h["source"] for h in hits],
		}
```

## Explanation

**The prompt is doing three specific jobs**, and each one prevents a real
failure:

1. **"using ONLY the context"** — stops the model answering from its general
   training knowledge. You want *your* documents' answer, not the internet's.
   Without this line, you can't tell which one you got.

2. **"If the context does not contain the answer, say I don't know"** — this is
   your hallucination guard. Remember: retrieval *always* returns its top 3,
   even for the pizza question. This sentence gives the model explicit
   permission to refuse. Without it, a model handed 3 irrelevant chunks will
   usually try to be helpful and invent something.

3. **"Cite the source filename"** — this is the payoff for carrying `source`
   through every single stage since `loader.py`. Now a human can verify the
   answer instead of trusting it.

**Why chunks are wrapped in `[filename]` markers.** The model can't see your
data structures — it only sees one flat block of text. Without markers, three
chunks run together into an indistinguishable wall and the model can't tell you
which fact came from where. The markers make citation possible.

**`sources` is returned separately** from the answer text so your app can show
clickable links, without having to parse them back out of English prose.

## Setup

```bash
.venv/bin/pip install anthropic
export ANTHROPIC_API_KEY="your-key-here"
```

Note: `requirements.txt` currently lists `openai`. Swap it for `anthropic`, or
keep both if you want to try each.

## Test it

```python
from src.rag import RAG

rag = RAG()

result = rag.ask("How do I build a web API in Python?")
print(result["answer"])
print("sources:", result["sources"])

# The honesty test - this is the one that matters
result = rag.ask("What is the best pizza recipe?")
print(result["answer"])   # should say "I don't know"
```

**That second test is the real measure of your system.** A RAG system that
answers everything confidently is broken. One that knows the limits of its
documents is working.

---

## When it's all running, experiment

The code is a starting point. Understanding comes from breaking it:

1. **Remove `normalize_embeddings=True`** and re-run stage 4. Results still
   appear — now subtly misordered. See the silent failure for yourself.

2. **Set `chunk_size=10`.** Watch precision improve and coherence collapse as
   sentences get shredded mid-thought.

3. **Set `top_k=1`, then `top_k=10`.** Find where extra context starts hurting.

4. **Add a long document** (a few pages) so chunking finally does real work.
   Right now your files are ~35 words each, so each becomes a single chunk and
   the overlap logic never activates.

5. **Ask something answerable only by combining two documents** — e.g. *"Can I
   store embeddings for a FastAPI app?"* (needs `postgres.txt` + `fastapi.txt`).
   Multi-hop questions are where basic RAG starts to strain, and where
   techniques like reranking and query rewriting come in.

## Where to go after this

- **Persistence** — right now everything rebuilds on startup. `faiss.write_index`
  saves to disk.
- **Hybrid search** — combine vector search with keyword search. Vectors are bad
  at exact matches like error codes and product IDs; keywords handle those.
- **Reranking** — retrieve 20, then use a slower, more accurate model to pick
  the best 3.
- **Evaluation** — write question/expected-source pairs and measure how often
  the right document ranks first. Without measurement, you're guessing.
  
  echo 'sk-1234efgh5678ijkl1234efgh5678ijkl1234efgh' > .env
  echo 'OPENAI_API_KEY=sk-1234efgh5678ijkl1234efgh5678ijkl1234efgh' > .env

