Retrieval on 99 questions (test), 42,838 chunks in the index. The model is handed 8 pages.

| Search | Right filing in top 8 | Evidence page first | In top 5 | In top 8 | Evidence pages found in top 8 | MRR |
|---|---|---|---|---|---|---|
| bm25 | 63.6% | 9.1% | 14.1% | 18.2% | 18.2% | 0.123 |
| vector | 90.9% | 16.2% | 44.4% | 56.6% | 52.5% | 0.293 |
| hybrid | 84.8% | 12.1% | 37.4% | 42.4% | 41.4% | 0.228 |
| hybrid+rerank | 89.9% | 18.2% | 41.4% | 55.6% | 51.5% | 0.287 |
| bm25, routed | 96.0% | 13.1% | 33.3% | 39.4% | 36.9% | 0.224 |
| vector, routed | 98.0% | 24.2% | 57.6% | 67.7% | 63.6% | 0.394 |
| hybrid, routed | 97.0% | 22.2% | 51.5% | 62.6% | 58.1% | 0.361 |
| hybrid+rerank, routed | 99.0% | 24.2% | 50.5% | 62.6% | 58.6% | 0.365 |
| bm25, routed, statements first | 98.0% | 21.2% | 57.6% | 67.7% | 66.7% | 0.385 |
| vector, routed, statements first | 98.0% | 24.2% | 63.6% | 76.8% | 75.8% | 0.424 |
| hybrid, routed, statements first | 98.0% | 24.2% | 64.6% | 75.8% | 75.8% | 0.425 |
| hybrid+rerank, routed, statements first | 99.0% | 25.3% | 61.6% | 76.8% | 76.8% | 0.429 |
