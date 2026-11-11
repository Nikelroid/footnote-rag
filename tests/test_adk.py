"""The ADK version of the agent: how it is put together, and that its model calls are counted. No model is called."""

import types as pytypes
from datetime import date

import pytest
from google.adk.models import Gemini

import footnote.app.adk_agent as adk_agent
from footnote import config, pricing
from footnote.ledger import BudgetExceeded, Ledger
from footnote.llm import MissingApiKey, load_google_key
from footnote.pricing import UnknownModel
from test_app import stub


def request(text="What was capex?"):
    part = pytypes.SimpleNamespace(text=text, function_call=None, function_response=None)
    return pytypes.SimpleNamespace(
        config=pytypes.SimpleNamespace(system_instruction="Be brief.", tools=[]), contents=[pytypes.SimpleNamespace(parts=[part])]
    )


def response(prompt=1200, output=200, thoughts=50, partial=False, cached=0):
    usage = pytypes.SimpleNamespace(
        prompt_token_count=prompt, candidates_token_count=output, thoughts_token_count=thoughts, cached_content_token_count=cached
    )
    return pytypes.SimpleNamespace(partial=partial, usage_metadata=usage)


@pytest.fixture
def on_2026_11_10(monkeypatch):
    """The date is pinned, so that a price with an end date keeps meaning the same thing in these tests."""
    monkeypatch.setattr(pricing, "_today", lambda: date(2026, 11, 10))


def test_agent_has_the_search_tool_and_the_budget_callbacks():
    agent = adk_agent.build_agent(ledger=Ledger(":memory:"))
    assert agent.name == "footnote" and agent.tools == [adk_agent.search_filings]
    assert agent.model.model == "anthropic/claude-haiku-5-5"
    assert agent.before_model_callback and agent.after_model_callback and agent.on_model_error_callback
    assert "search_filings" in agent.instruction and "Final answer:" in agent.instruction


def test_the_tool_returns_pages_and_calls_no_model(monkeypatch):
    one = stub()
    monkeypatch.setattr(adk_agent, "_assistant", lambda: one)
    result = adk_agent.search_filings("3M 2018 purchases of property, plant and equipment")
    assert result["pages"][0]["page"] == 59 and one.llm.calls == []


def test_every_model_call_is_reserved_then_settled_at_its_real_cost():
    ledger = Ledger(":memory:")
    guard = adk_agent.LedgerGuard(ledger, "anthropic/claude-haiku-5-5")
    guard.before(None, request())
    assert ledger.reserved() > 0 and ledger.spent() == 0
    guard.after(None, response(partial=True))  # a streamed piece changes nothing
    assert ledger.reserved() > 0
    guard.after(None, response())
    assert ledger.reserved() == 0 and ledger.spent() == pytest.approx((1200 * 0.10 + 250 * 0.50) / 1e6)  # thinking is billed as output


def test_a_call_past_the_cap_is_refused_and_a_failed_call_stays_counted():
    ledger = Ledger(":memory:", cap_usd=0.0015, lines={"development": 0.0015})
    guard = adk_agent.LedgerGuard(ledger, "anthropic/claude-haiku-5-5")
    guard.before(None, request())
    guard.error(None, request(), RuntimeError("connection lost"))
    assert ledger.health()["unconfirmed_calls"] == 1 and ledger.spent() > 0
    with pytest.raises(BudgetExceeded):
        guard.before(None, request())


def test_a_model_with_no_price_is_refused_before_anything_runs():
    with pytest.raises(UnknownModel):
        adk_agent.build_agent(model="gemini-9-flash", ledger=Ledger(":memory:"))


def test_the_same_agent_runs_on_gemini_through_google_cloud(on_2026_11_10):
    ledger = Ledger(":memory:", cap_usd=1.0, lines={"adk": 1.0})
    agent = adk_agent.build_agent(model=adk_agent.GEMINI, ledger=ledger, api_key="test-key")
    assert isinstance(agent.model, Gemini) and agent.model.model == "gemini-3.8-flash"
    assert agent.model.client_kwargs == {"enterprise": True, "api_key": "test-key"}
    assert agent.tools == [adk_agent.search_filings] and agent.before_model_callback.__self__.ledger is ledger


def test_a_gemini_call_is_priced_from_its_own_row_and_cached_input_is_cheaper(on_2026_11_10):
    ledger = Ledger(":memory:", cap_usd=1.0, lines={"adk": 1.0})
    guard = adk_agent.LedgerGuard(ledger, "gemini-3.8-flash", line="adk")
    guard.before(None, request())
    guard.after(None, response(prompt=1200, output=200, thoughts=50, cached=1000))
    assert ledger.reserved() == 0 and ledger.spent() == pytest.approx((200 * 1.50 + 1000 * 0.15 + 250 * 7.50) / 1e6)


def test_gemini_spending_goes_to_its_own_ledger_with_its_own_cap(on_2026_11_10, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GOOGLE_LEDGER_PATH", tmp_path / "ledger-google.sqlite")
    monkeypatch.setattr(config, "LEDGER_PATH", tmp_path / "ledger.sqlite")
    guard = adk_agent.build_agent(model=adk_agent.GEMINI).before_model_callback.__self__
    assert guard.ledger.path == str(tmp_path / "ledger-google.sqlite") and guard.ledger.cap_usd == config.GOOGLE_CAP_USD
    guard.before(None, request())
    assert guard.ledger.reserved() > 0 and not (tmp_path / "ledger.sqlite").exists()  # the Claude budget is not touched


def test_a_price_past_its_end_date_is_refused(monkeypatch):
    rates = pricing.Rates(input=1.0, cache_write_5m=1.0, cache_write_1h=1.0, cache_read=0.1, output=5.0)
    monkeypatch.setitem(pricing.PRICING, "gemini-9-trial", pricing.ModelPricing(base=rates, valid_until=date(2026, 12, 31)))
    monkeypatch.setattr(pricing, "_today", lambda: date(2026, 12, 31))
    assert pricing.resolve_model("gemini-9-trial") == "gemini-9-trial"  # still good on its last day
    monkeypatch.setattr(pricing, "_today", lambda: date(2027, 1, 1))
    with pytest.raises(UnknownModel):
        adk_agent.build_agent(model="gemini-9-trial", ledger=Ledger(":memory:"))
    assert pricing.resolve_model("claude-haiku-5-5") == "claude-haiku-5-5"  # a price with no end date still works


def test_the_google_key_is_read_from_the_env_file(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("GOOGLE_API_KEY=test-google-key\nGOOGLE_GENAI_USE_ENTERPRISE=false\n")
    monkeypatch.setattr(config, "ENV_FILE", env)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert load_google_key() == "test-google-key" and adk_agent._cloud_endpoint() is False
    env.write_text("ANTHROPIC_API_KEY=x\n")
    assert adk_agent._cloud_endpoint() is True  # Google Cloud's endpoint is the default
    with pytest.raises(MissingApiKey):
        load_google_key()


def test_the_command_stops_with_a_plain_message_when_the_google_key_is_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert adk_agent.main(["--gemini", "What", "was", "capex?"]) == 2
    assert "GOOGLE_API_KEY" in capsys.readouterr().out
