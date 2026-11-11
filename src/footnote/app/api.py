"""A small web API around the assistant.

    uvicorn footnote.app.api:app

    POST /search  {"query": "..."}        the best pages. Free, no model call.
    POST /ask     {"question": "..."}     an answer with the pages it cites. Uses the API key.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..ledger import BudgetExceeded
from ..llm import MissingApiKey, SpendLimitReached

app = FastAPI(title="Footnote", description="Question answering over SEC filings, with the page each answer comes from.")


class SearchBody(BaseModel):
    query: str = Field(min_length=3)
    pages: int = Field(default=5, ge=1, le=20)


class AskBody(BaseModel):
    question: str = Field(min_length=3)
    agent: bool = True


@lru_cache(maxsize=1)
def assistant():
    """Loaded once, on the first request. Opening the index and the models takes a few seconds."""
    from ..rag.assistant import Assistant

    return Assistant()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search")
def search(body: SearchBody) -> dict:
    return {"query": body.query, "pages": assistant().search(body.query, pages=body.pages)}


@app.post("/ask")
def ask(body: AskBody) -> dict:
    try:
        return assistant().ask(body.question, agent=body.agent)
    except MissingApiKey as err:
        raise HTTPException(status_code=503, detail=str(err)) from err
    except (BudgetExceeded, SpendLimitReached) as err:
        raise HTTPException(status_code=402, detail=str(err)) from err
