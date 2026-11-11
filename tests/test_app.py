"""The assistant, the API and the MCP tool, with stand-ins for search and the model."""

import asyncio

import pytest
from fastapi.testclient import TestClient

import footnote.app.api as api
import footnote.app.mcp_server as mcp_server
from footnote.llm import MissingApiKey
from footnote.rag.assistant import Assistant
from test_rag import BETTER, QUESTION, FakeRetriever, model


def stub() -> Assistant:
    return Assistant(retriever=FakeRetriever(), llm=model())


def test_ask_with_the_agent_finds_and_cites_the_page():
    r = stub().ask(QUESTION)
    assert r["found"] is True and r["final_answer"] == "$1,577 million"
    assert r["citations"] == [{"doc": "3M_2018_10K", "page": 59}]
    assert (r["searches"], r["model_calls"]) == (2, 2) and r["cost_usd"] > 0
    assert r["pages"][0]["page"] == 59 and r["pages"][0]["text"].startswith("Whole page 59.")


def test_ask_without_the_agent_says_not_found():
    r = stub().ask(QUESTION, agent=False)
    assert r["found"] is False and r["final_answer"] == "NOT FOUND" and (r["searches"], r["model_calls"]) == (1, 1)


def test_search_needs_no_api_key():
    assistant = Assistant(retriever=FakeRetriever())
    pages = assistant.search(BETTER, pages=1)
    assert [p["page"] for p in pages] == [59] and assistant._llm is None  # the client was never created
    assert assistant.retriever.searched_in == [["3M_2018_10K"]]  # the query names 3M, so only its filing is searched


@pytest.fixture
def client(monkeypatch):
    one = stub()
    monkeypatch.setattr(api, "assistant", lambda: one)
    return TestClient(api.app)


def test_api_routes(client):
    assert client.get("/health").json() == {"status": "ok"}
    found = client.post("/search", json={"query": BETTER, "pages": 2})
    assert found.status_code == 200 and found.json()["pages"][0]["doc"] == "3M_2018_10K"
    answer = client.post("/ask", json={"question": QUESTION})
    assert answer.status_code == 200 and answer.json()["final_answer"] == "$1,577 million"
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/search", json={"query": BETTER, "pages": 99}).status_code == 422


def test_api_says_so_when_the_key_is_missing(monkeypatch):
    class NoKey(Assistant):
        @property
        def llm(self):
            raise MissingApiKey("No API key found.")

    monkeypatch.setattr(api, "assistant", lambda: NoKey(retriever=FakeRetriever()))
    reply = TestClient(api.app).post("/ask", json={"question": QUESTION})
    assert reply.status_code == 503 and "No API key" in reply.json()["detail"]


def test_mcp_server_offers_the_search_tool(monkeypatch):
    one = stub()
    monkeypatch.setattr(mcp_server, "assistant", lambda: one)
    pages = mcp_server.search_filings(BETTER, pages=1)
    assert pages[0]["page"] == 59 and one.llm.calls == []  # the tool searches. It never calls a model.
    tools = asyncio.run(mcp_server.server.list_tools())
    assert [t.name for t in tools] == ["search_filings"] and "capital expenditure" in tools[0].description


def test_demo_page_loads_without_touching_the_index():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    page = AppTest.from_file(str(Path(__file__).parents[1] / "src" / "footnote" / "app" / "ui.py")).run(timeout=60)
    assert not page.exception and page.title[0].value == "Footnote"
    assert [t.label for t in page.toggle] == ["Search again when the first search is weak", "Search only, no model call"]
    assert [t.label for t in page.tabs] == ["Ask a question", "When a filing changes"]


def test_demo_page_shows_what_one_amended_line_is_answered_with(monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    import footnote.llm as llm_module
    from footnote.data.financebench import default_root
    from footnote.ledger import Ledger
    from footnote.llm import FakeLLM

    if not (default_root() / "data" / "financebench_open_source.jsonl").exists():
        pytest.skip("benchmark data not downloaded")

    def first_amount_wins(req):  # a stand-in model that reads the line on the first page it is given
        body = req.messages[-1]["content"]
        line = body.split("'")[1]
        amount = body.split(line, 1)[1].split("\n")[1:6]
        printed = next(cell.strip().strip("()$ ") for cell in amount if any(ch.isdigit() for ch in cell))
        return '{"reasoning": "Read from the first page.", "found": true, "final_answer": "' + printed + '", "pages": []}'

    monkeypatch.setattr(llm_module, "client_for", lambda model: FakeLLM(responder=first_amount_wins, ledger=Ledger(":memory:")))
    page = AppTest.from_file(str(Path(__file__).parents[1] / "src" / "footnote" / "app" / "ui.py")).run(timeout=60)
    assert len(page.selectbox[0].options) >= 40 and [r.label for r in page.radio] == ["What the update left in the index", "Which version the search returns first"]
    page.radio[0].set_value("appended")
    next(b for b in page.button if b.label == "Ask about this line").click().run(timeout=60)
    assert not page.exception and "This is a stale answer" in page.error[0].value  # both versions kept, old page first
    page.radio[0].set_value("synced")
    next(b for b in page.button if b.label == "Ask about this line").click().run(timeout=60)
    assert not page.exception and "gives the new amount" in page.success[0].value
    assert len(page.table) == 1  # the two lines the amendment changed
