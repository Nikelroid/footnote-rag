import pytest

from footnote.ledger import Ledger
from footnote.llm import FakeLLM, LLMRequest, ServedModelMismatch
from footnote.pricing import PROMPT_PAD_TOKENS, Usage, cost_usd, worst_case_cost_usd


def request(text="What was capex?", model="claude-haiku-5-5", **kw):
    return LLMRequest(model=model, messages=[{"role": "user", "content": text}], max_tokens=1000, **kw)


def test_success_settles_the_real_cost_on_the_right_line():
    llm = FakeLLM(responder=lambda r: "Final answer: 1,577", ledger=Ledger(":memory:"))
    response = llm.complete(request(), line="development", run_id="r1", purpose="answer")
    assert response.cost_usd == pytest.approx(cost_usd("claude-haiku-5-5", response.usage))
    assert llm.ledger.spent("development") == pytest.approx(response.cost_usd) and llm.ledger.reserved() == 0
    assert response.to_record()["model"] == "claude-haiku-5-5" and response.attempts == 1


def test_failure_releases_the_reservation():
    llm = FakeLLM(responder=lambda r: "x", ledger=Ledger(":memory:"), fail_times=1)
    with pytest.raises(ConnectionError):
        llm.complete(request(), line="development")
    assert llm.ledger.spent() == 0 and llm.ledger.reserved() == 0


def test_wrong_model_is_billed_then_rejected():
    llm = FakeLLM(responder=lambda r: "x", ledger=Ledger(":memory:"), served_model="claude-sonnet-5-5")
    with pytest.raises(ServedModelMismatch):
        llm.complete(request(), line="development")
    assert llm.ledger.spent() > 0


def test_snapshot_of_the_requested_model_is_accepted():
    llm = FakeLLM(responder=lambda r: "x", ledger=Ledger(":memory:"), served_model="claude-haiku-5-5-20261001")
    assert llm.complete(request(), line="development").model == "claude-haiku-5-5-20261001"


def test_api_arguments():
    kwargs = request(system="Be brief.", effort="low", json_schema={"type": "object"}).api_kwargs()
    assert kwargs["system"] == "Be brief."
    assert kwargs["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert "thinking" not in kwargs and "temperature" not in kwargs
    assert request(thinking="disabled").api_kwargs()["thinking"] == {"type": "disabled"}


def test_invalid_request_fails_before_money_is_reserved():
    llm = FakeLLM(responder=lambda r: "x", ledger=Ledger(":memory:"))
    with pytest.raises(ValueError):
        llm.complete(request(model="claude-sonnet-5-5", thinking="disabled"), line="grading")
    assert llm.calls == [] and llm.ledger.reserved() == 0


class Counting(FakeLLM):
    counted = 0

    def exact_prompt_tokens(self, req):
        Counting.counted += 1
        return 120_000  # over the long-context line, although the character guess says otherwise


def test_large_prompts_are_counted_exactly_before_reserving():
    cap = worst_case_cost_usd("claude-haiku-5-5", 0, 1000, prompt_tokens=120_000)
    # The worst case prices every prompt token at the one-hour cache-write rate, on the long-context tier.
    assert cap == pytest.approx(((120_000 + PROMPT_PAD_TOKENS) * 1.00 + 1000 * 2.50) / 1e6)
    assert cap >= cost_usd("claude-haiku-5-5", Usage(input_tokens=120_000, output_tokens=1000))
    llm = Counting(responder=lambda r: "x", ledger=Ledger(":memory:", cap_usd=cap * 0.9, lines={"comparison": cap}))
    big = request("word " * 15_000)  # 75,000 characters: under 100,000 tokens by the guess of 1.5 characters per token
    from footnote.ledger import BudgetExceeded

    with pytest.raises(BudgetExceeded):
        llm.complete(big, line="comparison")
    assert Counting.counted == 1 and llm.calls == []
    small = FakeLLM(responder=lambda r: "x", ledger=Ledger(":memory:"))
    small.complete(request(), line="development")  # a small prompt is never counted


def test_only_measurable_prompts_are_accepted():
    blocks = [{"type": "text", "text": "abc"}, {"type": "text", "text": "defg"}]
    assert LLMRequest(model="m", messages=[{"role": "user", "content": blocks}], max_tokens=1, system="xy").prompt_chars() == 9
    cached = [{"type": "text", "text": "a" * 300_000, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
    with pytest.raises(ValueError):
        LLMRequest(model="m", messages=[{"role": "user", "content": "q"}], max_tokens=1, system=cached).prompt_chars()
    document = [{"type": "document", "source": {"type": "text", "data": "a" * 300_000}}]
    llm = FakeLLM(responder=lambda r: "x", ledger=Ledger(":memory:"))
    with pytest.raises(ValueError):
        llm.complete(LLMRequest(model="claude-haiku-5-5", messages=[{"role": "user", "content": document}], max_tokens=1), line="development")
    assert llm.calls == [] and llm.ledger.reserved() == 0


def test_worst_case_covers_a_prompt_written_to_the_one_hour_cache():
    # 90,000 tokens written to the one-hour cache is the dearest way to send a prompt under the long-context line.
    real = cost_usd("claude-haiku-5-5", Usage(cache_write_1h_tokens=90_000, output_tokens=500))
    assert worst_case_cost_usd("claude-haiku-5-5", 0, 500, prompt_tokens=90_000) >= real


def test_failure_after_sending_keeps_the_money_counted():
    llm = FakeLLM(responder=lambda r: "x", ledger=Ledger(":memory:"), fail_after_send_times=1)
    with pytest.raises(ConnectionError):
        llm.complete(request(), line="development")
    reserved_for_it = worst_case_cost_usd("claude-haiku-5-5", request().prompt_chars(), 1000)
    assert llm.ledger.spent() == pytest.approx(reserved_for_it) and llm.ledger.reserved() == 0
    assert llm.ledger.health()["unconfirmed_calls"] == 1 and llm.ledger.health()["overrun_calls"] == 0
