"""One door to the Claude API.

Every request in the project goes through `LLM.complete`. It reserves budget
before the request, records the real token counts and cost afterwards, retries
transient errors with jittered backoff, and checks that the model which answered
is the model that was asked for.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Callable

from . import config
from .ledger import Ledger, google_ledger
from .pricing import (
    EXACT_COUNT_ABOVE,
    UnknownModel,
    Usage,
    cost_usd,
    prompt_token_bound,
    resolve_model,
    worst_case_cost_usd,
)

API_BASE_URL = "https://api.anthropic.com"
MAX_ATTEMPTS = 5


class MissingApiKey(RuntimeError):
    pass


class ServedModelMismatch(RuntimeError):
    pass


class SpendLimitReached(RuntimeError):
    """The account's own spend limit stopped the request. Never retried."""


@dataclass
class LLMRequest:
    model: str
    messages: list[dict]
    max_tokens: int
    system: str | None = None
    effort: str | None = None  # low | medium | high | xhigh | max. None keeps the model default.
    thinking: str | None = None  # None keeps the model default (adaptive). "disabled" turns it off.
    json_schema: dict | None = None  # constrain the reply to this JSON schema

    def prompt_chars(self) -> int:
        """Size of everything sent. Anything that cannot be measured is refused, because it could not be budgeted."""
        if self.system is not None and not isinstance(self.system, str):
            raise ValueError("system must be a plain string, so that its size can be bounded")
        total = len(self.system or "")
        for message in self.messages:
            content = message["content"]
            if isinstance(content, str):
                total += len(content)
                continue
            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)):
                    raise ValueError("only text can be sent through this client. Other content has no size bound here")
                total += len(block["text"])
        if self.json_schema is not None:
            total += len(json.dumps(self.json_schema))
        return total

    def api_kwargs(self) -> dict:
        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens, "messages": self.messages}
        if self.system:
            kwargs["system"] = self.system
        output_config: dict = {}
        if self.effort:
            output_config["effort"] = self.effort
        if self.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": self.json_schema}
        if output_config:
            kwargs["output_config"] = output_config
        if self.thinking == "disabled":
            # Claude Sonnet 5.5 and Claude Opus 5.5 reject a request that disables thinking.
            if resolve_model(self.model) != "claude-haiku-5-5":
                raise ValueError(f"thinking cannot be disabled on {self.model}")
            kwargs["thinking"] = {"type": "disabled"}
        elif self.thinking is not None:
            raise ValueError(f"unsupported thinking setting {self.thinking!r}")
        return kwargs


@dataclass
class LLMResponse:
    text: str
    stop_reason: str | None
    model: str  # the model that served the request, read from the response
    usage: Usage
    thinking_tokens: int
    cost_usd: float
    latency_s: float  # the final, successful attempt only
    attempts: int
    request_id: str | None

    def to_record(self) -> dict:
        return {
            "model": self.model,
            "stop_reason": self.stop_reason,
            "usage": self.usage.to_dict(),
            "thinking_tokens": self.thinking_tokens,
            "cost_usd": self.cost_usd,
            "latency_s": round(self.latency_s, 3),
            "attempts": self.attempts,
            "request_id": self.request_id,
        }


@dataclass
class _Raw:
    text: str
    stop_reason: str | None
    model: str
    usage: Usage
    thinking_tokens: int = 0
    request_id: str | None = None
    latency_s: float = 0.0
    attempts: int = 1
    unconfirmed_attempts: int = 0  # earlier attempts that failed after they were sent


class LLM:
    """Budget and bookkeeping shared by the real client and the test double."""

    def __init__(self, ledger: Ledger | None):
        self.ledger = ledger

    def _send(self, req: LLMRequest) -> _Raw:
        """Send the request. An exception may carry `unconfirmed_attempts`, the number of failed attempts that could have been billed."""
        raise NotImplementedError

    def exact_prompt_tokens(self, req: LLMRequest) -> int | None:
        """The true prompt size, when the client can count it without paying. None means unknown."""
        return None

    def complete(self, req: LLMRequest, *, line: str, run_id: str | None = None, purpose: str | None = None) -> LLMResponse:
        req.api_kwargs()  # an invalid request fails here, before any money is reserved
        chars = req.prompt_chars()
        call_id, worst = None, 0.0
        if self.ledger is not None:
            exact = self.exact_prompt_tokens(req) if prompt_token_bound(chars) > EXACT_COUNT_ABOVE else None
            worst = worst_case_cost_usd(req.model, chars, req.max_tokens, prompt_tokens=exact)
            call_id = self.ledger.reserve(line, req.model, worst, run_id=run_id, purpose=purpose)

        def charge_unconfirmed(count: int) -> None:
            for _ in range(count):
                self.ledger.add_unconfirmed(line, req.model, worst, run_id=run_id, purpose=purpose)

        try:
            raw = self._send(req)
        except BaseException as err:
            if call_id is not None:
                unconfirmed = getattr(err, "unconfirmed_attempts", 0)
                if unconfirmed:
                    # Sent, then failed. It may have been billed, so the money stays counted.
                    self.ledger.keep_unconfirmed(call_id)
                    charge_unconfirmed(unconfirmed - 1)
                else:
                    self.ledger.release(call_id)
            raise
        try:
            cost, priced = cost_usd(raw.model, raw.usage), True
        except UnknownModel:
            # The reply names a model with no price. Bill it at the requested model's price, then reject it below.
            cost, priced = cost_usd(req.model, raw.usage), False
        if call_id is not None:
            self.ledger.settle(call_id, raw.model, raw.usage, cost, raw.request_id)
            charge_unconfirmed(raw.unconfirmed_attempts)
        if not priced or resolve_model(raw.model) != resolve_model(req.model):
            raise ServedModelMismatch(f"asked for {req.model}, served by {raw.model}")
        return LLMResponse(
            text=raw.text,
            stop_reason=raw.stop_reason,
            model=raw.model,
            usage=raw.usage,
            thinking_tokens=raw.thinking_tokens,
            cost_usd=cost,
            latency_s=raw.latency_s,
            attempts=raw.attempts,
            request_id=raw.request_id,
        )


def load_api_key() -> str:
    from dotenv import dotenv_values

    key = dotenv_values(config.ENV_FILE).get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise MissingApiKey(
            f"No API key found. Copy .env.example to {config.ENV_FILE} and paste your key after ANTHROPIC_API_KEY=."
        )
    return key


def load_google_key() -> str:
    """The Google Cloud key for the ADK agent on Gemini. Nothing else in the project uses it."""
    from dotenv import dotenv_values

    key = dotenv_values(config.ENV_FILE).get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise MissingApiKey(f"No Google key found. Paste it after GOOGLE_API_KEY= in {config.ENV_FILE}. docs/keys.md has the steps.")
    return key


class AnthropicLLM(LLM):
    def __init__(self, ledger: Ledger, api_key: str | None = None, timeout_s: float = 600.0):
        if ledger is None:
            raise ValueError("the real client never runs without a ledger")
        super().__init__(ledger)
        import anthropic

        self._anthropic = anthropic
        # The base URL is set here on purpose. The SDK would otherwise follow an
        # ANTHROPIC_BASE_URL variable from whatever shell started the run.
        # Retries are handled below so that every attempt is counted.
        self._client = anthropic.Anthropic(
            api_key=api_key or load_api_key(), base_url=API_BASE_URL, max_retries=0, timeout=timeout_s
        )

    def _retryable(self, err: Exception) -> bool:
        a = self._anthropic
        if isinstance(err, a.EnforcedSpendLimitReachedError):
            return False
        if isinstance(err, (a.RateLimitError, a.InternalServerError, a.APIConnectionError)):
            return True
        return isinstance(err, a.APIStatusError) and err.status_code in (408, 409)

    def _never_billed(self, err: Exception) -> bool:
        """True when the failed attempt cannot have been billed: the API answered with an error, or was never reached."""
        import httpx2

        if isinstance(err, self._anthropic.APIStatusError):
            return True
        return isinstance(err.__cause__, (httpx2.ConnectError, httpx2.ConnectTimeout))

    def _send(self, req: LLMRequest) -> _Raw:
        kwargs = req.api_kwargs()
        unconfirmed = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            start = time.monotonic()
            accepted = False
            try:
                with self._client.messages.stream(**kwargs) as stream:
                    accepted = True  # the API answered 200 and started the reply
                    message = stream.get_final_message()
                    request_id = stream.request_id
            except self._anthropic.EnforcedSpendLimitReachedError as err:
                raise SpendLimitReached(str(err)) from err
            except Exception as err:
                # A reply that broke off, or a request that vanished after it was sent, may
                # have been billed. Only an error status or a failed connection rules that out.
                if accepted or not self._never_billed(err):
                    unconfirmed += 1
                if attempt == MAX_ATTEMPTS or not self._retryable(err):
                    err.unconfirmed_attempts = unconfirmed
                    raise
                time.sleep(min(60.0, 2.0**attempt) + random.uniform(0, 1))
                continue
            latency = time.monotonic() - start
            u = message.usage
            write_5m = write_1h = 0
            if u.cache_creation is not None:
                write_5m = u.cache_creation.ephemeral_5m_input_tokens or 0
                write_1h = u.cache_creation.ephemeral_1h_input_tokens or 0
            elif u.cache_creation_input_tokens:
                write_5m = u.cache_creation_input_tokens
            details = u.output_tokens_details
            return _Raw(
                text="".join(block.text for block in message.content if block.type == "text"),
                stop_reason=message.stop_reason,
                model=message.model,
                usage=Usage(
                    input_tokens=u.input_tokens or 0,
                    output_tokens=u.output_tokens or 0,
                    cache_write_5m_tokens=write_5m,
                    cache_write_1h_tokens=write_1h,
                    cache_read_tokens=u.cache_read_input_tokens or 0,
                ),
                thinking_tokens=(details.thinking_tokens or 0) if details is not None else 0,
                request_id=request_id,
                latency_s=latency,
                attempts=attempt,
                unconfirmed_attempts=unconfirmed,
            )
        raise AssertionError("unreachable")

    def count_tokens(self, req: LLMRequest) -> int:
        """Exact prompt size for a request, from the token counting endpoint."""
        kwargs = {k: v for k, v in req.api_kwargs().items() if k in ("model", "messages", "system", "output_config", "thinking")}
        return self._client.messages.count_tokens(**kwargs).input_tokens

    def exact_prompt_tokens(self, req: LLMRequest) -> int | None:
        try:
            return self.count_tokens(req)
        except Exception:
            return None  # fall back to the character bound


def google_cloud_endpoint() -> bool:
    """True unless .env says the Google key was made in AI Studio. Only .env is read, never the shell."""
    from dotenv import dotenv_values

    return (dotenv_values(config.ENV_FILE).get("GOOGLE_GENAI_USE_ENTERPRISE") or "true").strip().lower() in ("true", "1")


# How a Gemini reply ended, in the words the rest of the project uses.
GOOGLE_STOP = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "refusal",
    "RECITATION": "refusal",
    "BLOCKLIST": "refusal",
    "PROHIBITED_CONTENT": "refusal",
    "SPII": "refusal",
}
GOOGLE_RETRY_CODES = (408, 429, 500, 502, 503, 504)


class GeminiLLM(LLM):
    """The same door for Gemini on Google Cloud, so that a second model family can answer the same questions.

    It is paid from Google Cloud credit and never from the Claude budget, so it
    is given the Google ledger, which has a cap of its own.
    """

    def __init__(self, ledger: Ledger, api_key: str | None = None, cloud: bool | None = None, timeout_s: float = 600.0):
        if ledger is None:
            raise ValueError("the real client never runs without a ledger")
        super().__init__(ledger)
        from google import genai
        from google.genai import errors, types

        self._types, self._errors = types, errors
        # The key is passed here on purpose. A key the library found in the shell would lose
        # to project settings found there. With no retry options the library never retries,
        # so every attempt is counted below.
        self._client = genai.Client(
            enterprise=google_cloud_endpoint() if cloud is None else cloud,
            api_key=api_key or load_google_key(),
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        )

    def _kwargs(self, req: LLMRequest) -> dict:
        t = self._types
        settings: dict = {"max_output_tokens": req.max_tokens}
        if req.system:
            settings["system_instruction"] = req.system
        if req.json_schema is not None:
            settings["response_mime_type"] = "application/json"
            settings["response_json_schema"] = req.json_schema
        if req.effort in ("low", "medium", "high"):
            settings["thinking_config"] = t.ThinkingConfig(thinking_level=req.effort.upper())
        contents = []
        for message in req.messages:
            content = message["content"]
            text = content if isinstance(content, str) else "".join(block["text"] for block in content)
            contents.append(t.Content(role="user" if message["role"] == "user" else "model", parts=[t.Part(text=text)]))
        return {"model": req.model, "contents": contents, "config": t.GenerateContentConfig(**settings)}

    def _send(self, req: LLMRequest) -> _Raw:
        kwargs = self._kwargs(req)
        unconfirmed = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            start = time.monotonic()
            try:
                response = self._client.models.generate_content(**kwargs)
            except Exception as err:
                # An error status from the API means nothing was billed. Anything else may have been sent.
                answered = isinstance(err, self._errors.APIError)
                if not answered:
                    unconfirmed += 1
                if attempt == MAX_ATTEMPTS or (answered and err.code not in GOOGLE_RETRY_CODES):
                    err.unconfirmed_attempts = unconfirmed
                    raise
                time.sleep(min(60.0, 2.0**attempt) + random.uniform(0, 1))
                continue
            latency = time.monotonic() - start
            u = response.usage_metadata
            if u is None:
                err = RuntimeError("the reply carried no token counts, so it cannot be priced")
                err.unconfirmed_attempts = unconfirmed + 1  # it was answered, so it was billed. The worst case stays counted.
                raise err
            candidate = response.candidates[0] if response.candidates else None
            parts = candidate.content.parts if candidate is not None and candidate.content and candidate.content.parts else []
            finish = candidate.finish_reason.name if candidate is not None and candidate.finish_reason else None
            # Gemini counts the part of the prompt it read from its cache inside the prompt total. That part is cheaper.
            cached = u.cached_content_token_count or 0
            thinking = u.thoughts_token_count or 0
            return _Raw(
                text="".join(p.text for p in parts if p.text and not p.thought),
                # No candidate at all means the prompt itself was blocked.
                stop_reason=GOOGLE_STOP.get(finish, finish.lower() if finish else "refusal"),
                model=response.model_version or req.model,
                usage=Usage(
                    input_tokens=max((u.prompt_token_count or 0) - cached, 0),
                    output_tokens=(u.candidates_token_count or 0) + thinking,  # thinking is billed as output
                    cache_read_tokens=cached,
                ),
                thinking_tokens=thinking,
                request_id=response.response_id,
                latency_s=latency,
                attempts=attempt,
                unconfirmed_attempts=unconfirmed,
            )
        raise AssertionError("unreachable")


def client_for(model: str) -> LLM:
    """The paid client for a model, with the ledger its spending belongs in."""
    if model.startswith("gemini-"):
        return GeminiLLM(google_ledger())
    return AnthropicLLM(Ledger())


@dataclass
class FakeLLM(LLM):
    """A deterministic stand-in for tests and dry runs. It costs nothing and still feeds the ledger."""

    responder: Callable[[LLMRequest], str] = lambda req: ""
    ledger: Ledger | None = None
    stop_reason: str = "end_turn"
    served_model: str | None = None
    fail_times: int = 0  # fail before anything is sent
    fail_after_send_times: int = 0  # fail after the request went out
    calls: list[LLMRequest] = field(default_factory=list)

    def _send(self, req: LLMRequest) -> _Raw:
        self.calls.append(req)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("simulated failure before sending")
        if self.fail_after_send_times > 0:
            self.fail_after_send_times -= 1
            err = ConnectionError("simulated failure after sending")
            err.unconfirmed_attempts = 1
            raise err
        text = self.responder(req)
        return _Raw(
            text=text,
            stop_reason=self.stop_reason,
            model=self.served_model or req.model,
            usage=Usage(input_tokens=req.prompt_chars() // 3 + 1, output_tokens=len(text) // 3 + 1),
            latency_s=0.001,
        )
