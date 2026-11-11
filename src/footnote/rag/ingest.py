"""Step 1. Turn filing PDFs into something searchable.

    PDF  ->  pages  ->  chunks  ->  vectors in Chroma
                 \\->  whole pages kept in a small page store

Four choices matter here.

1. A chunk never crosses a page. The page is what an answer cites, and an edit
   on one page then changes the chunks of that page only.
2. Every chunk starts with a line that says where it comes from, such as
   "3M | 10k | 2018 | page 60". Search can then tell one company's cash flow
   statement from another's.
3. Small chunks are good for searching and bad for reading, because a line from
   the middle of a table has lost its column headers. So search runs on chunks
   and the model is given the whole page a chunk came from.
4. Indexing goes through LangChain's indexing API. It remembers a hash of every
   chunk, so running it again embeds only what changed and removes what is gone.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.indexing import index
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config
from ..data.financebench import FETCH_HINT, MissingData, default_root, load_documents

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
INDEX_DIR = config.DATA_DIR / "index"


def pdf_path(doc_name: str) -> Path:
    return default_root() / "pdfs" / f"{doc_name}.pdf"


def load_pages(doc_name: str, info: dict, path: Path | None = None) -> list[Document]:
    """One document per page, tagged with the filing it belongs to. Pages are numbered from 0, as in the benchmark."""
    path = Path(path) if path else pdf_path(doc_name)
    if not path.exists():
        raise MissingData(f"{path} is not there yet. {FETCH_HINT}")
    pages = []
    with pymupdf.open(path) as pdf:
        for number, page in enumerate(pdf):
            text = page.get_text().strip()
            if not text:
                continue
            metadata = {
                "source": doc_name,
                "page": number,
                "company": info.get("company", ""),
                "doc_type": info.get("doc_type", ""),
                "period": str(info.get("doc_period", "")),
            }
            pages.append(Document(page_content=text, metadata=metadata))
    return pages


def split(pages: list[Document]) -> list[Document]:
    """Cut pages into chunks. Each page is split on its own, so no chunk runs over a page break."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        m = chunk.metadata
        header = f"{m['company']} | {m['doc_type']} | {m['period']} | page {m['page'] + 1}"
        chunk.page_content = f"{header}\n{chunk.page_content}"
    return chunks


def embeddings() -> HuggingFaceEmbeddings:
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, model_kwargs={"device": device}, encode_kwargs={"normalize_embeddings": True, "batch_size": 64}
    )


class PageStore:
    """Whole pages, looked up by filing and page number. Search finds a chunk, the model reads its page."""

    def __init__(self, path: Path):
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS pages (source TEXT, page INTEGER, text TEXT, PRIMARY KEY (source, page))")

    def replace(self, source: str, pages: list[Document]) -> None:
        with self._db:  # one transaction, so a filing is never half replaced
            self._db.execute("DELETE FROM pages WHERE source = ?", (source,))
            self._db.executemany(
                "INSERT INTO pages VALUES (?, ?, ?)", [(source, p.metadata["page"], p.page_content) for p in pages]
            )

    def get(self, source: str, page: int) -> str | None:
        row = self._db.execute("SELECT text FROM pages WHERE source = ? AND page = ?", (source, page)).fetchone()
        return row[0] if row else None

    def of(self, source: str) -> dict[int, str]:
        """Every page of one filing, by page number."""
        return dict(self._db.execute("SELECT page, text FROM pages WHERE source = ?", (source,)))

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]


class Index:
    """Everything search needs: the vector store, the record of what is in it, and the pages."""

    def __init__(self, collection: str = "filings", directory: Path | None = None):
        from langchain_classic.indexes import SQLRecordManager

        self.directory = directory or INDEX_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            collection_name=collection, embedding_function=embeddings(), persist_directory=str(self.directory / "chroma")
        )
        self.records = SQLRecordManager(f"chroma/{collection}", db_url=f"sqlite:///{self.directory / 'records.sqlite'}")
        self.records.create_schema()
        self.pages = PageStore(self.directory / "pages.sqlite")

    def sync(self, doc_name: str, pages: list[Document]) -> dict:
        """Bring the index in step with this version of one filing.

        Unchanged chunks are skipped. New and changed chunks are embedded. Chunks of
        this filing that are no longer there are deleted, so old text cannot be found.
        Returns the counts: num_added, num_updated, num_skipped, num_deleted.

        The cleanup mode is "scoped_full". The better known "incremental" mode cleans
        up after every batch, so a filing with more chunks than one batch would have
        its later chunks deleted and written again on every run.
        """
        result = index(
            split(pages), self.records, self.store, cleanup="scoped_full", source_id_key="source", batch_size=256, key_encoder="sha256"
        )
        self.pages.replace(doc_name, pages)
        return dict(result)


def ingest(doc_names: list[str] | None = None, log=print) -> dict:
    """Index the filings behind the benchmark questions, or the ones named."""
    table = load_documents()
    if doc_names is None:
        from ..data.financebench import load_questions

        doc_names = sorted({q.doc_name for q in load_questions()})
    idx = Index()
    totals = {"num_added": 0, "num_updated": 0, "num_skipped": 0, "num_deleted": 0}
    for i, name in enumerate(doc_names, 1):
        result = idx.sync(name, load_pages(name, table.get(name, {})))
        for key in totals:
            totals[key] += result[key]
        log(f"[{i}/{len(doc_names)}] {name}: {result}")
    totals["pages"] = idx.pages.count()
    return totals
