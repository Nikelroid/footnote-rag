"""Step 2. Find the pages that answer a question.

First the question is routed: if it names a company and a year, only those
filings are searched (see `route.py`).

Then two searches run side by side, because they fail in different ways.

- Keyword search (BM25) matches exact words and numbers, such as "purchases of
  property, plant and equipment". It misses a question that uses other words.
- Vector search matches meaning. It can miss an exact name or figure.

A cross-encoder then reads every candidate from both next to the question and
puts them in order. It is slower than the first two, so it only sees the few
dozen candidates they return. Without it, the two rankings are merged with
reciprocal rank fusion.

Search works on chunks. What is handed on is the whole page of each good chunk,
with repeats removed, because a page keeps the table headers a chunk has lost.
The main financial statements of the routed filing are put first, because
search misses them too often (see `statements.py`).

LangChain's EnsembleRetriever does the same fusion. It is written out here
because the method is ten lines and worth knowing.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .ingest import Index
from .route import route
from .statements import main_statements

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODES = ("bm25", "vector", "hybrid", "hybrid+rerank")

# Words that say nothing about which page is wanted. They are dropped from the keyword query only.
STOPWORDS = frozenset(
    "a an and are as at be been by can could did do does for from had has have how i if in into is it its of on or our "
    "should than that the their them then there these they this those to was we were what when where which who why will "
    "with would you your answer question response please give provide using based relying details shown following "
    "information state calculate assume".split()
)


@dataclass
class Hit:
    doc: str  # the filing
    page: int  # numbered from 0
    text: str
    score: float
    filed: str | None = None  # the date the filing was filed, when the index knows it

    @property
    def chunk_id(self) -> str:
        return hashlib.sha256(f"{self.doc}|{self.page}|{self.text}".encode()).hexdigest()[:16]

    def to_record(self) -> dict:
        return {"doc": self.doc, "page": self.page, "chunk_id": self.chunk_id, "score": round(float(self.score), 4)}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def fuse(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion. An item scores 1 / (k + rank) in every list that has it, and the scores add up."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self, index: Index, candidates: int = 30, filed: dict[str, str] | None = None):
        self.index = index
        self.candidates = candidates
        # Filing dates by filing name. With them every page handed to the model carries its date, and the
        # answer step is told that the later filing wins. That matters once two versions of a filing are indexed.
        self.filed = dict(filed or {})
        self.chunks: list[Document] = []
        while True:  # read the store in slices. Chroma refuses to return tens of thousands of chunks in one call.
            data = index.store.get(include=["documents", "metadatas"], limit=5000, offset=len(self.chunks))
            self.chunks += [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
            if len(data["documents"]) < 5000:
                break
        self._tokens = [tokens(c.page_content) for c in self.chunks]
        self._by_source: dict[str, list[int]] = {}
        self.filings: dict[str, dict] = {}  # what routing needs to know about each filing
        for i, chunk in enumerate(self.chunks):
            m = chunk.metadata
            self._by_source.setdefault(m["source"], []).append(i)
            self.filings.setdefault(m["source"], {"company": m["company"], "doc_period": m["period"]})
        self._bm25: dict[tuple, tuple[BM25Okapi, list[int]]] = {}
        self._statements: dict[str, list[int]] = {}
        self._reranker = None
        self._models = threading.Lock()  # the two neural models are used by one thread at a time

    def route(self, question: str) -> list[str] | None:
        return route(question, self.filings)

    @staticmethod
    def _hit(doc: Document, score: float) -> Hit:
        return Hit(doc=doc.metadata["source"], page=int(doc.metadata["page"]), text=doc.page_content, score=score)

    def _keyword_index(self, sources: list[str] | None) -> tuple[BM25Okapi, list[int]]:
        """BM25 over the chunks being searched. Built on first use and kept, since a few filings are searched again and again."""
        key = tuple(sorted(sources)) if sources else ()
        with self._models:
            if key not in self._bm25:
                ids = [i for s in key for i in self._by_source.get(s, [])] if key else list(range(len(self.chunks)))
                self._bm25[key] = (BM25Okapi([self._tokens[i] for i in ids]), ids)
            return self._bm25[key]

    def keyword(self, query: str, sources: list[str] | None = None) -> list[Hit]:
        bm25, ids = self._keyword_index(sources)
        words = [w for w in tokens(query) if w not in STOPWORDS]
        scores = bm25.get_scores(words)
        best = np.argsort(-scores)[: self.candidates]
        return [self._hit(self.chunks[ids[i]], float(scores[i])) for i in best if scores[i] > 0]

    def vector(self, query: str, sources: list[str] | None = None) -> list[Hit]:
        where = None
        if sources:
            where = {"source": sources[0]} if len(sources) == 1 else {"source": {"$in": list(sources)}}
        with self._models:
            found = self.index.store.similarity_search_with_score(query, k=self.candidates, filter=where)
        return [self._hit(doc, -float(distance)) for doc, distance in found]

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return []
        with self._models:
            if self._reranker is None:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(RERANK_MODEL)
            scores = self._reranker.predict([(query, h.text) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda pair: -pair[1])
        return [Hit(h.doc, h.page, h.text, float(s)) for h, s in ranked]

    def search(self, query: str, mode: str = "hybrid+rerank", sources: list[str] | None = None) -> list[Hit]:
        """Chunks, best first. With `sources`, only those filings are searched."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "bm25":
            return self.keyword(query, sources)
        if mode == "vector":
            return self.vector(query, sources)
        by_id, rankings = {}, []
        for hits in (self.keyword(query, sources), self.vector(query, sources)):
            rankings.append([h.chunk_id for h in hits])
            by_id.update({h.chunk_id: h for h in hits})
        merged = [by_id[key] for key in fuse(rankings)]
        return merged if mode == "hybrid" else self.rerank(query, merged)

    def pinned(self, sources: list[str] | None) -> list[Hit]:
        """The main financial statements of the filings a question was routed to. Nothing when it was routed to many."""
        if not sources or len(sources) > 2:
            return []
        out = []
        for source in sources:
            if source not in self._statements:
                self._statements[source] = main_statements(self.index.pages.of(source))
            out += [Hit(source, page, self.index.pages.get(source, page), 0.0, self.filed.get(source)) for page in self._statements[source]]
        return out

    def pages(self, hits: list[Hit], n: int, first: list[Hit] | None = None) -> list[Hit]:
        """Whole pages for the model: the `first` pages, then the page behind each chunk, each page once."""
        out = list(first or [])[:n]
        seen = {(p.doc, p.page) for p in out}
        for hit in hits:
            if len(out) >= n:
                break
            key = (hit.doc, hit.page)
            if key in seen:
                continue
            seen.add(key)
            text = self.index.pages.get(hit.doc, hit.page)
            out.append(Hit(hit.doc, hit.page, text if text is not None else hit.text, hit.score, self.filed.get(hit.doc)))
        return out

    def find(self, question: str, mode: str = "hybrid+rerank", n_pages: int = 8, routed: bool = True, pin: bool = True) -> list[Hit]:
        """The pages to read for a question: route, search, then put the main statements first."""
        sources = self.route(question) if routed else None
        return self.pages(self.search(question, mode, sources), n_pages, first=self.pinned(sources) if pin else None)
