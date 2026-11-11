# When the filings change

A RAG system is usually built and tested on a frozen set of documents. Real document sets do not stay frozen, and filings are a clear case. This file describes what goes wrong when part of the corpus keeps changing, what Footnote does about it, and how that is tested.

## How filings change

| Change | Example | What the index must do |
|---|---|---|
| A new document arrives | The next quarter's 10-Q | Add it without touching the rest |
| A document is amended | A 10-K/A that corrects the 10-K | Replace the old text, and keep it for history |
| A number is restated | Last year's figure, changed in this year's filing | Answer with the right version for the date asked |
| A document is withdrawn | A filing pulled or superseded | Stop returning it in search |

## What goes wrong without a plan

- **Stale answers.** The old chunk stays in the index next to the new one, and the system answers with the outdated number as confidently as before. A 2025 benchmark built for this problem, HoH, found that outdated information in the retrieval source lowers answer accuracy ([arXiv 2503.04800](https://arxiv.org/abs/2503.04800)).
- **Full rebuilds.** Re-parsing and re-embedding everything after each change is slow, and with a paid embedding or context step it is also the largest cost.
- **Half-applied updates.** A reader that searches while a document is being replaced sees a mix of old and new text.
- **No history.** After the old text is deleted, nobody can say what the system would have answered last month, or which version an old answer came from.

## What Footnote does

The registry in `src/footnote/corpus/registry.py` is a record of which text is searchable at which time. It was built first, as a component with its own tests. The running assistant takes a simpler route, described further down.

1. **Check the file first.** A hash of the source file is stored with each version. If the hash has not changed, the document is skipped before any parsing.
2. **Identify chunks by their text.** A chunk's id is a hash of its text with whitespace normalised. Text that did not change keeps its id and its embedding. Only text never seen before is embedded, and this holds across versions and across documents, so boilerplate shared by many filings is embedded once.
3. **Keep edits local.** Chunks never cross a page. An edit on one page then changes the chunks of that page and no others. With fixed-size chunks, one inserted sentence moves every later boundary.
4. **Close old versions instead of deleting them.** Each version has a `valid_from` and a `valid_to` date. Search over current text reads only versions that are still open, so a retired chunk cannot be returned. A question about a past date reads the versions that were open on that date.
5. **Apply an update in one transaction.** A document is replaced completely or not at all. A failed update leaves the registry exactly as it was.
6. **Record the version in the citation.** An answer cites the document, the version and the page, so an old answer can be traced to the text it came from.

Points 1 to 5 are built and tested in the registry. Point 6 is not built. The running assistant does not use the registry. It keeps its index in step with LangChain's indexing API, described below, which skips the chunks that did not change and deletes the ones that are gone. Its answers cite the document and the page.

Values that change every day, such as a share price, do not belong in an index at all. They are better fetched live by a tool at question time.

### Measured on a synthetic filing

A synthetic 40-page filing was indexed, then one sentence was inserted on page 10 and the filing was indexed again.

| Chunking | Chunks | Embedded again | Share |
|---|---|---|---|
| Fixed size, 1,000 characters with 100 overlap | 128 | 98 | 76.6% |
| Page-anchored | 81 | 3 | 3.7% |

Another test in the same file replaces one number in a small filing. Search over current text then returns the new number and never the old one, and a search as of the earlier date still returns the old one. The numbers come from `tests/test_corpus.py`.

### On a real filing

The running system uses LangChain's indexing API for this (`src/footnote/rag/ingest.py`, method `Index.sync`). `footnote updates-demo` changes one amount on one page of the 3M 2018 annual report and syncs it again.

| Action | Chunks embedded | Skipped | Deleted | Time |
|---|---|---|---|---|
| First index | 635 | 0 | 0 | 4.04 s |
| After the amendment | 1 | 634 | 1 | 0.34 s |
| The same version again | 0 | 635 | 0 | 0.09 s |

After the update no search returned the old amount, and the amended line kept its place near the top of the results. One detail matters. The cleanup mode has to be "scoped_full". The "incremental" mode cleans up after every batch, and a filing with more chunks than one batch then has its later chunks deleted and written again on every run.

### Relation to LangChain's indexing API

LangChain's `index` function covers the first two jobs. It stores a hash of each document with a record manager, skips what is unchanged, and its cleanup modes delete what is no longer in the source ([reference](https://reference.langchain.com/python/langchain-core/indexing/api)). Because it deletes the text it replaces, it keeps no history. The registry keeps old versions, since some questions are about a past date and an old answer should stay traceable.

## What the assistant answers after an amendment

Keeping the index in step is half of the problem. The other half is what the model does when the index is not in step, and both versions of a page reach it. That was measured, and the full tables are in [experiments.md](experiments.md). The short version follows.

**The test.** Take a real statement page. Move an amount from one line to another line under the same total, as a reclassification does in an amended filing. Every total still adds up, and the new amounts exist nowhere else, so the model cannot know them from training. Then ask for the changed line. An answer with the old amount is a stale answer. No judge is needed, because both amounts are known.

**Why the amount is moved and not just changed.** The first version of the test changed one amount and nothing else. That breaks a total on the page, and the models noticed. They added up the lines and sided with the version that added up, which was the old one. That is a finding of its own, but it tests arithmetic and not updating. `src/footnote/corpus/sums.py` finds which lines add up to which totals, and uses that to build an amendment that keeps every total intact.

**What was found, with Claude Haiku 5.5 on 45 lines, three runs each.**

| What the update left | Answers that give the old amount, alone or first |
|---|---|
| Only the amended page (the index was synced) | 0.0% |
| Both pages, old one first | 87.4% |
| Both pages, new one first | 0.0% |
| Both pages, each with its filing date, old one first | 100.0% |
| Both pages, each with its filing date, new one first | 38.5% |
| Both pages with dates, and one sentence that the later filing wins | 0.0% |

The model always noticed that two versions disagreed. It still led with one of them, and the search order decided which. Filing dates alone made it worse, because the model tended to treat the earlier filing as the original. One sentence of instruction fixed it.

**The same test through the real index.** The page-level test hands the model its pages directly. `src/footnote/rag/amended_index.py` puts the amended filing into a real index next to the old one, as a 10-K/A would arrive, and sends the question through routing, search and the answer step. With both filings in the index the assistant gave the old amount, alone or first, in 97.0% of answers. With the fix switched on, which is the filing date on every page plus the one sentence, it gave the new amount in 97.0%. With the old filing replaced it gave the new amount in 99.0%.

**What follows for a real system.**

- Delete or replace on update. This is what `Index.sync` does.
- If both versions have to stay, store the filing date, show it on every page, and tell the model which version wins. `Retriever(index, filed=...)` switches this on in Footnote.
- Do not rely on the model to work out which version is current. It trusts the first page it reads, and it trusts the page whose numbers add up.

**Real amendments.** The amendments in this test are made up, on real pages. A small set of real amended filings from SEC EDGAR would be the next check.

## Status

| Part | State |
|---|---|
| Versioned registry, change sets, as-of reads | Built as a separate component, 9 tests |
| Fixed-size and page-anchored chunkers | Built, 4 tests |
| Index that follows an amended filing | Built, measured on a real filing |
| Amendments that keep the page's totals intact | Built, 45 cases from 32 filings |
| Stale and fresh answer rates, page by page | Measured on three models |
| The same through the real index, with the fix | Measured |
