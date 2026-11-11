"""Step 4. What happens when a filing changes.

A filing gets amended, a number is restated, a new quarter arrives. If the index
does not follow, the assistant keeps answering with the old text, and it sounds
just as sure as before.

The fix has two parts, and both are in `ingest.Index.sync`.

1. Know what is already indexed. LangChain's record manager stores a hash of
   every chunk, grouped by the filing it came from.
2. On the next sync, compare. A chunk whose hash is known is skipped. A new hash
   is embedded and written. A hash that belongs to the same filing and did not
   come back is deleted, so the old text can no longer be found.

Because chunks never cross a page (see `ingest.split`), an edit on one page
touches only that page's chunks. This file runs that end to end on a scratch
index and reports what it cost.
"""

from __future__ import annotations

import shutil
import time

from langchain_core.documents import Document

from ..corpus.edits import _factor, _format, _on_page
from ..corpus.probes import clean_lines
from ..data.financebench import load_documents
from . import ingest
from .retrieve import Retriever


def pick_line(pages: list[Document]) -> tuple[int, str, str]:
    """A labelled statement line whose amount is printed on one page only: (position in the list, label, amount)."""
    for i, page in enumerate(pages):
        for label, amount in clean_lines(page.page_content):
            if not any(_on_page(amount).search(p.page_content) for j, p in enumerate(pages) if j != i):
                return i, label, amount
    raise ValueError("no line with a unique amount in this filing")


def amend(pages: list[Document], position: int, amount: str, new_amount: str) -> list[Document]:
    """A second version of the filing in which one amount on one page is different."""
    out = list(pages)
    old = pages[position]
    out[position] = Document(page_content=_on_page(amount).sub(new_amount, old.page_content), metadata=dict(old.metadata))
    return out


def _timed(index: ingest.Index, doc_name: str, pages: list[Document]) -> dict:
    start = time.monotonic()
    counts = index.sync(doc_name, pages)
    return {**counts, "seconds": round(time.monotonic() - start, 2)}


def demo(doc_name: str = "3M_2018_10K") -> dict:
    scratch = ingest.INDEX_DIR / "updates_demo"
    shutil.rmtree(scratch, ignore_errors=True)  # a scratch index, so the real one is never touched
    index = ingest.Index(collection="updates_demo", directory=scratch)
    info = load_documents()[doc_name]
    pages = ingest.load_pages(doc_name, info)
    position, label, amount = pick_line(pages)
    page_number = pages[position].metadata["page"]
    new_amount = _format(round(float(amount.replace(",", "")) * _factor(doc_name)), 0, "," in amount)
    amended = amend(pages, position, amount, new_amount)
    query = f"{info['company']} {label}"

    def rank(retriever: Retriever, printed: str) -> int | None:
        """Where the chunk that prints this amount comes in the search results. None when no result prints it."""
        hits = retriever.search(query, mode="hybrid")
        return next((i for i, h in enumerate(hits, start=1) if _on_page(printed).search(h.text)), None)

    first = _timed(index, doc_name, pages)
    rank_before = rank(Retriever(index), amount)
    second = _timed(index, doc_name, amended)
    retriever = Retriever(index)
    third = _timed(index, doc_name, amended)

    result = {
        "filing": doc_name,
        "line": label,
        "page": page_number,
        "old_amount": amount,
        "new_amount": new_amount,
        "first_index": first,
        "after_amendment": second,
        "same_version_again": third,
        "share_embedded_again": round(second["num_added"] / max(1, first["num_added"]), 4),
        "rank_of_the_line_before": rank_before,
        "rank_of_the_line_after": rank(retriever, new_amount),
        "old_amount_still_found_by_search": rank(retriever, amount) is not None,
        "chunks_still_holding_old_amount": sum(1 for c in retriever.chunks if _on_page(amount).search(c.page_content)),
        "page_store_has_new_amount": bool(_on_page(new_amount).search(index.pages.get(doc_name, page_number) or "")),
    }
    shutil.rmtree(scratch, ignore_errors=True)
    return result
