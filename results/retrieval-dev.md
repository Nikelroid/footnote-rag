Retrieval on 51 questions (dev), 42,838 chunks in the index. The model is handed 8 pages.

| Search | Right filing in top 8 | Evidence page first | In top 5 | In top 8 | Evidence pages found in top 8 | MRR |
|---|---|---|---|---|---|---|
| bm25 | 62.7% | 5.9% | 11.8% | 11.8% | 11.8% | 0.087 |
| vector | 92.2% | 21.6% | 37.3% | 49.0% | 43.1% | 0.310 |
| hybrid | 84.3% | 9.8% | 29.4% | 37.3% | 33.3% | 0.191 |
| hybrid+rerank | 92.2% | 15.7% | 43.1% | 51.0% | 46.1% | 0.266 |
| bm25, routed | 96.1% | 11.8% | 31.4% | 37.3% | 34.3% | 0.223 |
| vector, routed | 96.1% | 27.5% | 54.9% | 64.7% | 58.5% | 0.411 |
| hybrid, routed | 98.0% | 23.5% | 51.0% | 56.9% | 51.0% | 0.355 |
| hybrid+rerank, routed | 98.0% | 17.6% | 54.9% | 62.7% | 57.8% | 0.332 |
| bm25, routed, statements first | 98.0% | 27.5% | 72.5% | 74.5% | 72.5% | 0.451 |
| vector, routed, statements first | 96.1% | 29.4% | 74.5% | 80.4% | 78.4% | 0.467 |
| hybrid, routed, statements first | 98.0% | 27.5% | 70.6% | 76.5% | 74.5% | 0.450 |
| hybrid+rerank, routed, statements first | 98.0% | 25.5% | 72.5% | 84.3% | 81.4% | 0.451 |
