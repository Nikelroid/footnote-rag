"""Step 6. The changed-filing test, through the whole pipeline.

`stale.py` hands the model its pages directly, which shows what the model does
with two versions of a page. This file asks the question the way a user would.
The amended filing goes into a real index, and the question travels the real
path: route, search, pin the statements, answer.

A scratch index holds each filing twice. The filing as first filed keeps its
name, such as 3M_2018_10K. The amended filing is added under the name a 10-K/A
would get, 3M_2018_10KA. The same index is then read in four ways.

    before     only the filings as first filed. A case counts only if the pipeline answers it here.
    synced     the amended filing has replaced the old one.
    appended   both are in the index. Nothing but the name tells them apart.
    fixed      both are in the index, every page carries its filing date, and the
               answer step is told that the later filing wins. This is the fix
               the page-level test pointed to, switched on in the real assistant.

The question no longer names a file. It names the company and the year, as a
user would, so routing has to find the filings by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from langchain_core.documents import Document

from .. import config
from ..corpus.edits import _on_page
from ..data.financebench import Question, load_documents
from ..systems.base import RunContext, SystemOutput, fingerprint
from . import ingest
from .graph import MODE, PAGES, answer_request, read_reply
from .retrieve import Hit, Retriever
from .stale import RULES, Case, build_cases, changes

SUFFIX = "A"  # form 10-K/A amends form 10-K, so 3M_2018_10K becomes 3M_2018_10KA
VIEWS = ("before", "synced", "appended", "fixed")
DIRECTORY = ingest.INDEX_DIR / "amended"


def _question(case: Case) -> str:
    q = case.question
    return f"In {q.company}'s annual report for fiscal {q.doc_period}, what is the first amount on the line '{q.evidence[0].text}'? Answer with the number as printed."


def amend_filing(pages: list[Document], cases: list[Case]) -> tuple[list[Document], list[Case]]:
    """The filing with every case's amounts changed, under its new name, and the cases that could be applied.

    The index holds the text PyMuPDF reads from the PDF, which is laid out differently from the benchmark's
    page text. A case is applied only when each old amount is printed exactly once on its PDF page.
    """
    position = {p.metadata["page"]: i for i, p in enumerate(pages)}
    texts = {i: p.page_content for i, p in enumerate(pages)}
    applied = []
    for case in cases:
        i = position.get(case.page)
        pairs = [(before, after) for _, before, after in changes(case)]
        if i is None or not all(len(_on_page(old).findall(texts[i])) == 1 and not _on_page(new).search(texts[i]) for old, new in pairs):
            continue
        for old, new in pairs:
            texts[i] = _on_page(old).sub(new, texts[i])
        applied.append(case)
    name = pages[0].metadata["source"] + SUFFIX
    return [Document(page_content=texts[i], metadata={**p.metadata, "source": name}) for i, p in enumerate(pages)], applied


def build(questions: list[Question], directory=None, log=print) -> tuple[ingest.Index, list[Case]]:
    """Index every annual report that has a moved-amount case, as first filed and as amended. Returns the index and the cases.

    Running it again embeds nothing, because the index remembers what it holds.
    """
    table = load_documents()
    by_filing: dict[str, list[Case]] = {}
    taken = set()
    for case in build_cases(questions, "moved"):
        if case.question.doc_type == "10k" and (case.doc_name, case.page) not in taken:  # one case a page, so two edits never meet
            taken.add((case.doc_name, case.page))
            by_filing.setdefault(case.doc_name, []).append(case)
    index = ingest.Index(collection="amended", directory=directory or DIRECTORY)
    cases = []
    for n, (name, wanted) in enumerate(sorted(by_filing.items()), 1):
        pages = ingest.load_pages(name, table.get(name, {}))
        amended, applied = amend_filing(pages, wanted)
        if not applied:
            continue
        first, second = index.sync(name, pages), index.sync(name + SUFFIX, amended)
        log(f"[{n}/{len(by_filing)}] {name}: {len(applied)} cases, embedded {first['num_added'] + second['num_added']} chunks")
        cases += [replace(c, question=replace(c.question, question=_question(c)), group="index") for c in applied]
    return index, cases


@dataclass
class ThroughIndex:
    """The assistant's own pipeline over an index that holds an amended filing, read in one of four ways."""

    retriever: Retriever = field(repr=False)
    cases: dict[str, Case] = field(repr=False)
    view: str = "appended"
    model: str = config.ANSWER_MODEL
    retrieves: bool = False

    def __post_init__(self):
        if self.view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")

    @property
    def name(self) -> str:
        return f"index-{self.view}"

    def describe(self) -> dict:
        page = Hit("SAMPLE_2020_10K", 1, "Capital expenditure 100", 1.0, "2021-02-15" if self.view == "fixed" else None)
        return {
            "name": self.name,
            "study": "stale",
            "edit": "index",
            "edit_rules": RULES["index"],
            "policy": self.view,
            "model": self.model,
            "pages": PAGES,
            "chunks": len(self.retriever.chunks),
            "request": fingerprint(answer_request(self.model, "sample", [page])),
        }

    def sources(self, question: str) -> list[str]:
        """The filings this view lets the search see, out of the ones the question was routed to."""
        routed = self.retriever.route(question)
        if not routed:
            raise RuntimeError("the question was not routed to any filing")
        amended = {name for name in routed if name.endswith(SUFFIX) and name[: -len(SUFFIX)] in self.retriever.filings}
        if self.view == "before":
            return [name for name in routed if name not in amended]
        if self.view == "synced":
            return [name for name in routed if name + SUFFIX not in amended]
        return routed

    def find(self, case: Case) -> list[Hit]:
        question = case.question.question
        sources = self.sources(question)
        pages = self.retriever.pages(self.retriever.search(question, MODE, sources), PAGES, first=self.retriever.pinned(sources))
        if self.view != "fixed":
            return pages
        first_filed, amended = case.filed
        return [replace(p, filed=amended if p.doc.endswith(SUFFIX) else first_filed) for p in pages]

    def plan(self, question: Question) -> list:
        case = self.cases[question.id]
        return [answer_request(self.model, case.question.question, self.find(case))]

    def answer(self, question: Question, ctx: RunContext) -> SystemOutput:
        case = self.cases[question.id]
        pages = self.find(case)
        reply = ctx.complete(answer_request(self.model, case.question.question, pages), "answer")
        text, _, cited = read_reply(reply, pages)
        return SystemOutput(answer=text, retrieved=[p.to_record() for p in pages], citations=cited)


def systems(retriever: Retriever, cases: list[Case], model: str = config.ANSWER_MODEL, views: tuple[str, ...] | None = None) -> list[ThroughIndex]:
    by_id = {c.question.id: c for c in cases}
    return [ThroughIndex(retriever, by_id, view, model) for view in views or VIEWS]
