# Evaluation results

42 answerable and 12 unanswerable questions over 37 documents (312 chunks of up to 200 words). Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.

## Retrieval by mode

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | --- | --- | --- | --- |
| dense | 86% | 93% | 100% | 0.911 |
| keyword | 86% | 95% | 98% | 0.915 |
| hybrid | 93% | 100% | 100% | 0.960 |

## Refusal threshold

DocSage refuses without calling the LLM when the closest chunk's cosine similarity is below the threshold. *Answered* is the share of answerable questions let through; *refused* is the share of unanswerable questions stopped.

| Threshold | Answered | Refused | Balanced |
| --- | --- | --- | --- |
| 0.20 | 100% | 42% | 71% |
| 0.25 | 100% | 67% | 83% |
| 0.30 | 100% | 75% | 88% |
| 0.35 (current) | 100% | 83% | 92% |
| 0.40 | 98% | 100% | 99% |
| 0.45 | 98% | 100% | 99% |
| 0.50 | 88% | 100% | 94% |
| 0.55 | 74% | 100% | 87% |

## Chunk size (hybrid)

| Words per chunk | Chunks | Hit@1 | Hit@5 | MRR |
| --- | --- | --- | --- | --- |
| 100 | 511 | 95% | 100% | 0.966 |
| 200 | 312 | 93% | 100% | 0.960 |
| 300 | 262 | 93% | 100% | 0.964 |

## Misses (hybrid, not in top 5)

None.
