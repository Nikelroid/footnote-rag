# Code tour

This page is a reading guide. It says where each part of the project lives and in which order the files are easiest to read. Every file starts with a short text that says what it does and why.

## Where things are

| Path | What is in it |
|---|---|
| `src/footnote/rag/` | The assistant. Ingest, routing, search, the agent, updates, and the tests on pages that mislead |
| `src/footnote/corpus/` | The code that builds amendments with a known answer, and a versioned registry with its chunkers that is tested on its own |
| `src/footnote/data/` | The download of the benchmark files, checked against a pinned commit, and their loader |
| `src/footnote/systems/` | The two reference systems. One gets no documents and one is handed the evidence page |
| `src/footnote/eval/` | Running and grading, the judge, the statistics, the reports and the charts |
| `src/footnote/app/` | The API, the demo page, the MCP server and the ADK agent |
| `src/footnote/llm.py`, `ledger.py`, `pricing.py` | The one function every model call goes through, and the spend ledger behind it |
| `src/footnote/cli.py` | The `footnote` command. `footnote --help` lists what it can do |
| `tests/` | 260 tests. None of them calls a model, and 9 need the downloaded benchmark files and skip without them |
| `results/` | Every table and chart quoted in the README, as Markdown, JSON and SVG |
| `manifests/` | The pinned commit of the benchmark, with the size and hash of each file |

## Reading order

About an hour and a half. The files marked Step 1 to Step 6 at the top follow the path of the project.

| Order | File | Lines | What to look at |
|---|---|---|---|
| 1 | `rag/ingest.py` | 158 | `split`, which keeps a chunk inside its page, and `Index.sync`, which embeds only what changed |
| 2 | `rag/route.py` and `rag/statements.py` | 69 and 44 | Two small rule-based steps. One picks the filing, one finds the three main statements |
| 3 | `rag/retrieve.py` | 194 | `fuse`, `search` and `pages`. Keyword and vector search, merged and reranked |
| 4 | `rag/graph.py` | 177 | The graph is built in the last 20 lines. Read those first, then the nodes |
| 5 | `rag/updates.py` | 97 | One filing is amended and the index follows |
| 6 | `corpus/sums.py` | 130 | How the code finds which lines add up to which totals, and moves an amount without breaking a total |
| 7 | `rag/stale.py` | 483 | The tests on pages that mislead. Read the text at the top, then `AfterAmendment.request` and `classify` |
| 8 | `rag/amended_index.py` | 161 | The amendment test through the real index |
| 9 | `app/api.py`, `app/mcp_server.py`, `app/ui.py`, `app/adk_agent.py` | 57, 42, 121, 199 | Four ways to serve the same assistant. `rag/assistant.py` is the one class behind the first three |

The evaluation code can be read after that, or treated as a tool.

| File | What it does |
|---|---|
| `eval/runner.py` | Answers the questions with one system, then grades them in a second pass. Both passes can be stopped and resumed |
| `eval/judge.py` | The rubric and the judge call |
| `eval/humans.py` | Scores the judge against the 2,400 hand-labelled answers of the benchmark |
| `eval/metrics.py` | The bootstrap interval and the paired permutation test |
| `eval/simulate.py` | Checks those statistics on made-up benchmarks where the truth is known |
| `eval/report.py` | Builds every table from the rows on disk |
| `eval/charts.py` | Draws the charts as plain SVG, with no plotting library |
| `llm.py` and `ledger.py` | Reserve the worst-case cost before a call, record the real cost after it, and refuse a call that would pass the cap |

[evaluation.md](evaluation.md) explains the design of the evaluation, [experiments.md](experiments.md) describes each experiment, and [changing-documents.md](changing-documents.md) covers the index side of a filing that changes.

## Free commands to start with

None of these calls a model. The first one downloads the benchmark files and the 84 filings, 170 MB, and the next three need them.

```bash
footnote data fetch --pdfs needed
footnote ingest               # about 6 minutes the first time, seconds after that
footnote retrieval-check      # what each search step adds
footnote updates-demo         # amend one filing and watch the index follow
footnote scorecard            # the table of pages that mislead, from the saved results
footnote stats-check          # the statistics, checked on simulated benchmarks
pytest                        # 260 tests
```
