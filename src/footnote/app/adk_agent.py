"""The same assistant on Google's Agent Development Kit (ADK).

    python -m footnote.app.adk_agent "What is the FY2018 capital expenditure for 3M?"
    python -m footnote.app.adk_agent --gemini "What is the FY2018 capital expenditure for 3M?"

LangGraph (`rag/graph.py`) spells the control flow out: route, search, check,
search again, answer. ADK works the other way round. The agent is given a tool
and instructions, and the model decides for itself when to search and when it
has enough. That is less code and less control. The graph can promise "at most
two searches". Here that promise is a sentence in the instructions and a limit
on the number of model calls.

The same agent runs on two models. As written it reaches Claude through
LiteLLM. With `--gemini` ADK talks to Gemini itself, through Google Cloud. That
run is paid from Google Cloud credit and not from the Claude budget, so it is
counted in a ledger of its own with a cap of its own. docs/keys.md says which
key each of the two needs.

ADK calls the model itself, so these calls do not pass through the project's
one door to the API. Two callbacks put them in the spend ledger anyway. One
reserves the worst-case cost before every model call and refuses the call if
the cap would be passed. The other records the real token counts afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from functools import lru_cache

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from .. import config
from ..ledger import Ledger, google_ledger
from ..pricing import Usage, cost_usd, resolve_model, worst_case_cost_usd

MODEL = "anthropic/claude-haiku-5-5"
GEMINI = config.GOOGLE_MODEL
MAX_OUTPUT_TOKENS = 2048
MAX_MODEL_CALLS = 6

INSTRUCTION = (
    "You answer questions about the financial filings of public companies. Always call search_filings before you answer, "
    "and use only the pages it returns. If the pages are not enough, search again with the words a filing would use for the "
    "statement or line item. Search at most three times. Name the document and page behind every number. If the pages do not "
    "contain the answer, say so and do not guess. End with one line that starts with 'Final answer:'."
)


@lru_cache(maxsize=1)
def _assistant():
    from ..rag.assistant import Assistant

    return Assistant()


def search_filings(query: str) -> dict:
    """Find the filing pages most relevant to a question.

    Args:
        query: What to look for. Name the company and the year.

    Returns:
        The best pages, each with its document name, its page number counted from 0, and its full text.
    """
    return {"pages": _assistant().search(query, pages=5)}


class LedgerGuard:
    """Counts every model call the agent makes against the project's budget."""

    def __init__(self, ledger: Ledger, model: str, line: str = "development"):
        self.ledger = ledger
        self.model = resolve_model(model.split("/")[-1])  # a model with no price in the table is refused here
        self.line = line
        self._open: list[int] = []

    @staticmethod
    def _chars(llm_request) -> int:
        config = llm_request.config
        total = len(str(getattr(config, "system_instruction", "") or "")) + len(str(getattr(config, "tools", "") or ""))
        for content in llm_request.contents or []:
            for part in content.parts or []:
                total += len(part.text or "") + len(str(part.function_call or "")) + len(str(part.function_response or ""))
        return total

    def before(self, context, llm_request):
        worst = worst_case_cost_usd(self.model, self._chars(llm_request), MAX_OUTPUT_TOKENS)
        self._open.append(self.ledger.reserve(self.line, self.model, worst, run_id="adk", purpose="adk"))
        return None  # None lets the call go ahead. A refused reservation raises instead.

    def after(self, context, llm_response):
        if getattr(llm_response, "partial", False) or not self._open:
            return None
        call_id = self._open.pop()
        usage = llm_response.usage_metadata
        if usage is None:
            self.ledger.keep_unconfirmed(call_id)  # no token counts came back, so the worst case stays counted
            return None
        # Gemini reports the part of the prompt it read from its cache. That part is cheaper.
        cached = getattr(usage, "cached_content_token_count", None) or 0
        spent = Usage(
            input_tokens=max((usage.prompt_token_count or 0) - cached, 0),
            cache_read_tokens=cached,
            output_tokens=(usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
        )
        self.ledger.settle(call_id, self.model, spent, cost_usd(self.model, spent))
        return None

    def error(self, context, llm_request, error):
        if self._open:
            self.ledger.keep_unconfirmed(self._open.pop())
        return None


def is_gemini(model: str) -> bool:
    return model.startswith("gemini-")


def build_agent(model: str = MODEL, ledger: Ledger | None = None, api_key: str | None = None, cloud: bool = True) -> Agent:
    if is_gemini(model):
        guard = LedgerGuard(ledger if ledger is not None else google_ledger(), model, line="adk")
        # `enterprise` sends the request to Google Cloud's endpoint, where Cloud credit applies.
        # A key made in AI Studio belongs to Google's other endpoint and needs cloud=False.
        llm = Gemini(model=model, client_kwargs={"enterprise": cloud, "api_key": api_key} if api_key else None)
    else:
        guard = LedgerGuard(ledger if ledger is not None else Ledger(), model)
        llm = LiteLlm(model=model, api_key=api_key) if api_key else LiteLlm(model=model)
    return Agent(
        name="footnote",
        description="Answers questions about SEC filings and cites the page.",
        model=llm,
        instruction=INSTRUCTION,
        tools=[search_filings],
        generate_content_config=types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS),
        before_model_callback=guard.before,
        after_model_callback=guard.after,
        on_model_error_callback=guard.error,
    )


async def _ask(question: str, agent: Agent) -> str:
    runner = InMemoryRunner(agent=agent, app_name="footnote")
    session = await runner.session_service.create_session(app_name="footnote", user_id="local")
    message = types.Content(role="user", parts=[types.Part(text=question)])
    answer = ""
    async for event in runner.run_async(
        user_id="local", session_id=session.id, new_message=message, run_config=RunConfig(max_llm_calls=MAX_MODEL_CALLS)
    ):
        if event.is_final_response() and event.content and event.content.parts:
            answer = "".join(part.text or "" for part in event.content.parts)
    return answer


def _cloud_endpoint() -> bool:
    """True unless .env says the Google key was made in AI Studio. Only .env is read, never the shell."""
    from ..llm import google_cloud_endpoint

    return google_cloud_endpoint()


def ask(question: str, model: str = MODEL) -> str:
    from ..llm import load_api_key, load_google_key

    if is_gemini(model):
        return asyncio.run(_ask(question, build_agent(model, api_key=load_google_key(), cloud=_cloud_endpoint())))
    # The shell may carry a base URL meant for another tool. It is removed for this
    # process, so the key is only ever sent to the real API.
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE"):
        os.environ.pop(name, None)
    return asyncio.run(_ask(question, build_agent(model, api_key=load_api_key())))


def main(argv: list[str] | None = None) -> int:
    from ..ledger import BudgetExceeded
    from ..llm import MissingApiKey
    from ..pricing import UnknownModel

    parser = argparse.ArgumentParser(prog="python -m footnote.app.adk_agent", description="Ask the ADK agent one question.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--gemini", action="store_true", help=f"run on {GEMINI} with the Google key, paid from Google Cloud credit")
    args = parser.parse_args(argv)
    try:
        print(ask(" ".join(args.question), GEMINI if args.gemini else MODEL))
    except (MissingApiKey, BudgetExceeded, UnknownModel) as err:
        print(f"Stopped: {err}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
