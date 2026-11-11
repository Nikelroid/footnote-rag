"""The real client, driven by a stand-in for the SDK's stream. No request leaves the machine."""

import types

import anthropic
import httpx2
import pytest

import footnote.llm as llm_module
from footnote.ledger import Ledger
from footnote.llm import API_BASE_URL, MAX_ATTEMPTS, AnthropicLLM, LLMRequest, SpendLimitReached
from footnote.pricing import Usage, cost_usd

REQUEST = httpx2.Request("POST", f"{API_BASE_URL}/v1/messages")


def status_error(cls, code):
    return cls("simulated", response=httpx2.Response(code, request=REQUEST), body=None)


def never_connected():
    err = anthropic.APIConnectionError(request=REQUEST)
    err.__cause__ = httpx2.ConnectError("connection refused")
    return err


class FakeStream:
    request_id = "req_123"  # the SDK reads this from the response headers, not from the message

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        if isinstance(self.message, Exception):
            raise self.message  # the reply started and then broke off
        return self.message


class BrokeOff:
    """A reply that the API accepted and that failed before it finished."""

    def __init__(self, error):
        self.error = error


def message(model="claude-haiku-5-5", stop_reason="end_turn", **usage):
    fields = dict(
        input_tokens=1200,
        output_tokens=300,
        cache_creation=None,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        output_tokens_details=types.SimpleNamespace(thinking_tokens=120),
    )
    fields.update(usage)
    return types.SimpleNamespace(
        content=[types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text="Final answer: 1,577")],
        stop_reason=stop_reason,
        model=model,
        usage=types.SimpleNamespace(**fields),
    )


def make(monkeypatch, behaviours):
    llm = AnthropicLLM(Ledger(":memory:"), api_key="not-a-real-key")
    sent = []

    def stream(**kwargs):
        sent.append(kwargs)
        step = behaviours.pop(0)
        if isinstance(step, Exception):
            raise step
        return FakeStream(step.error if isinstance(step, BrokeOff) else step)

    monkeypatch.setattr(llm._client.messages, "stream", stream)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)
    return llm, sent


def ask(**kw):
    return LLMRequest(model="claude-haiku-5-5", system="Be brief.", messages=[{"role": "user", "content": "What was capex?"}], max_tokens=1000, **kw)


def test_client_ignores_a_base_url_left_in_the_shell(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    llm = AnthropicLLM(Ledger(":memory:"), api_key="not-a-real-key")
    assert str(llm._client.base_url).startswith(API_BASE_URL) and llm._client.max_retries == 0
    with pytest.raises(ValueError):
        AnthropicLLM(None, api_key="not-a-real-key")


def test_reply_is_read_and_priced_from_the_api_usage(monkeypatch):
    llm, sent = make(monkeypatch, [message()])
    r = llm.complete(ask(effort="low", json_schema={"type": "object"}), line="grading", purpose="judge")
    assert r.text == "Final answer: 1,577" and r.stop_reason == "end_turn" and r.request_id == "req_123"
    assert r.usage == Usage(input_tokens=1200, output_tokens=300) and r.thinking_tokens == 120 and r.attempts == 1
    assert r.cost_usd == pytest.approx((1200 * 0.10 + 300 * 0.50) / 1e6)
    assert llm.ledger.spent("grading") == pytest.approx(r.cost_usd)
    assert sent[0]["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert sent[0]["system"] == "Be brief." and "temperature" not in sent[0]


def test_cache_tokens_are_split_by_lifetime(monkeypatch):
    split = types.SimpleNamespace(ephemeral_5m_input_tokens=4000, ephemeral_1h_input_tokens=1000)
    llm, _ = make(monkeypatch, [message(cache_creation=split, cache_creation_input_tokens=5000, cache_read_input_tokens=8000)])
    r = llm.complete(ask(), line="development")
    assert r.usage == Usage(1200, 300, cache_write_5m_tokens=4000, cache_write_1h_tokens=1000, cache_read_tokens=8000)
    assert r.cost_usd == pytest.approx(cost_usd("claude-haiku-5-5", r.usage))
    llm, _ = make(monkeypatch, [message(cache_creation_input_tokens=5000)])  # an older reply shape with no split
    assert llm.complete(ask(), line="development").usage.cache_write_5m_tokens == 5000


def test_transient_errors_are_retried_and_counted(monkeypatch):
    llm, sent = make(
        monkeypatch,
        [status_error(anthropic.RateLimitError, 429), status_error(anthropic.InternalServerError, 529), never_connected(), message()],
    )
    r = llm.complete(ask(), line="development")
    assert r.attempts == 4 and len(sent) == 4
    assert llm.ledger.spent() == pytest.approx(r.cost_usd) and llm.ledger.reserved() == 0


def test_a_bad_request_is_not_retried(monkeypatch):
    llm, sent = make(monkeypatch, [status_error(anthropic.BadRequestError, 400), message()])
    with pytest.raises(anthropic.BadRequestError):
        llm.complete(ask(), line="development")
    assert len(sent) == 1 and llm.ledger.spent() == 0 and llm.ledger.reserved() == 0


def test_the_account_spend_limit_stops_at_once(monkeypatch):
    llm, sent = make(monkeypatch, [status_error(anthropic.EnforcedSpendLimitReachedError, 429), message()])
    with pytest.raises(SpendLimitReached):
        llm.complete(ask(), line="development")
    assert len(sent) == 1 and llm.ledger.reserved() == 0


def test_persistent_overload_gives_up_after_the_last_attempt(monkeypatch):
    llm, sent = make(monkeypatch, [status_error(anthropic.OverloadedError, 529) for _ in range(MAX_ATTEMPTS + 2)])
    with pytest.raises(anthropic.OverloadedError):
        llm.complete(ask(), line="development")
    assert len(sent) == MAX_ATTEMPTS and llm.ledger.reserved() == 0


def worst(llm, req):
    from footnote.pricing import worst_case_cost_usd

    return worst_case_cost_usd(req.model, req.prompt_chars(), req.max_tokens)


def test_a_reply_that_breaks_off_stays_in_the_ledger(monkeypatch):
    llm, sent = make(monkeypatch, [BrokeOff(ConnectionResetError("reset by peer")), message()])
    with pytest.raises(ConnectionResetError):
        llm.complete(ask(), line="development")
    assert len(sent) == 1  # an unknown failure after sending is not blindly sent again
    assert llm.ledger.spent() == pytest.approx(worst(llm, ask())) and llm.ledger.reserved() == 0
    assert llm.ledger.health()["unconfirmed_calls"] == 1


def test_a_retry_after_a_broken_reply_is_charged_for_both(monkeypatch):
    llm, sent = make(monkeypatch, [BrokeOff(anthropic.APIConnectionError(request=REQUEST)), message()])
    r = llm.complete(ask(), line="development")
    assert r.attempts == 2 and len(sent) == 2
    assert llm.ledger.spent() == pytest.approx(r.cost_usd + worst(llm, ask()))
    assert llm.ledger.health()["unconfirmed_calls"] == 1


def test_timeouts_count_as_possibly_billed_on_every_attempt(monkeypatch):
    llm, sent = make(monkeypatch, [anthropic.APITimeoutError(request=REQUEST) for _ in range(MAX_ATTEMPTS)])
    with pytest.raises(anthropic.APITimeoutError):
        llm.complete(ask(), line="development")
    assert len(sent) == MAX_ATTEMPTS
    assert llm.ledger.spent() == pytest.approx(MAX_ATTEMPTS * worst(llm, ask()))
    assert llm.ledger.health()["unconfirmed_calls"] == MAX_ATTEMPTS


def test_a_refused_connection_costs_nothing(monkeypatch):
    llm, _ = make(monkeypatch, [never_connected() for _ in range(MAX_ATTEMPTS)])
    with pytest.raises(anthropic.APIConnectionError):
        llm.complete(ask(), line="development")
    assert llm.ledger.spent() == 0 and llm.ledger.reserved() == 0


def test_a_connection_lost_after_sending_is_counted(monkeypatch):
    lost = anthropic.APIConnectionError(request=REQUEST)
    lost.__cause__ = httpx2.RemoteProtocolError("server disconnected")
    llm, _ = make(monkeypatch, [lost, message()])
    r = llm.complete(ask(), line="development")
    assert r.attempts == 2 and llm.ledger.health()["unconfirmed_calls"] == 1


def test_a_reply_from_a_model_with_no_price_is_billed_then_rejected(monkeypatch):
    from footnote.llm import ServedModelMismatch

    llm, _ = make(monkeypatch, [message(model="claude-mystery-1")])
    with pytest.raises(ServedModelMismatch):
        llm.complete(ask(), line="development")
    assert llm.ledger.spent() == pytest.approx((1200 * 0.10 + 300 * 0.50) / 1e6) and llm.ledger.reserved() == 0
    llm, _ = make(monkeypatch, [message(model="claude-haiku-5-5@20261001")])  # a dated id of the right model
    assert llm.complete(ask(), line="development").model == "claude-haiku-5-5@20261001"
