"""The Gemini client, driven by a stand-in for the SDK call. No request leaves the machine."""

import types

import pytest
from google.genai import errors

import footnote.llm as llm_module
from footnote import config
from footnote.ledger import Ledger
from footnote.llm import MAX_ATTEMPTS, AnthropicLLM, GeminiLLM, LLMRequest, ServedModelMismatch, client_for
from footnote.pricing import Usage, cost_usd


def reply(text="Final answer: 1,577", finish="STOP", model="gemini-3.8-flash", prompt=1200, cached=0, out=300, thoughts=120, usage=True):
    parts = [types.SimpleNamespace(text="hidden reasoning", thought=True), types.SimpleNamespace(text=text, thought=None)]
    candidate = types.SimpleNamespace(content=types.SimpleNamespace(parts=parts), finish_reason=types.SimpleNamespace(name=finish))
    counts = types.SimpleNamespace(prompt_token_count=prompt, cached_content_token_count=cached, candidates_token_count=out, thoughts_token_count=thoughts)
    return types.SimpleNamespace(candidates=[candidate], usage_metadata=counts if usage else None, model_version=model, response_id="resp_1")


def api_error(code):
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": "simulated", "status": "SIMULATED"}})


def make(monkeypatch, behaviours):
    ledger = Ledger(":memory:", cap_usd=config.GOOGLE_CAP_USD, lines=config.GOOGLE_BUDGET_LINES)
    llm = GeminiLLM(ledger, api_key="not-a-real-key", cloud=True)
    sent = []

    def generate_content(**kwargs):
        sent.append(kwargs)
        step = behaviours.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    monkeypatch.setattr(llm._client.models, "generate_content", generate_content)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)
    return llm, sent


def ask(**kw):
    return LLMRequest(model="gemini-3.8-flash", system="Be brief.", messages=[{"role": "user", "content": "What was capex?"}], max_tokens=1000, **kw)


def test_the_request_is_translated_for_gemini(monkeypatch):
    llm, sent = make(monkeypatch, [reply()])
    schema = {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}
    llm.complete(ask(json_schema=schema, effort="low"), line="comparison")
    call = sent[0]
    assert call["model"] == "gemini-3.8-flash"
    assert [(c.role, c.parts[0].text) for c in call["contents"]] == [("user", "What was capex?")]
    cfg = call["config"]
    assert cfg.system_instruction == "Be brief." and cfg.max_output_tokens == 1000
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema == schema
    assert cfg.thinking_config.thinking_level.name == "LOW"


def test_a_reply_is_priced_with_thinking_as_output_and_cached_input_at_its_own_rate(monkeypatch):
    llm, _ = make(monkeypatch, [reply(prompt=1200, cached=1000, out=300, thoughts=120)])
    response = llm.complete(ask(), line="comparison")
    assert response.text == "Final answer: 1,577"  # the model's hidden reasoning is not part of the answer
    assert response.usage == Usage(input_tokens=200, cache_read_tokens=1000, output_tokens=420) and response.thinking_tokens == 120
    assert response.cost_usd == pytest.approx((200 * 1.50 + 1000 * 0.15 + 420 * 7.50) / 1e6)
    assert response.cost_usd == pytest.approx(cost_usd("gemini-3.8-flash", response.usage))
    assert response.stop_reason == "end_turn" and response.model == "gemini-3.8-flash" and response.request_id == "resp_1"
    assert llm.ledger.spent("comparison") == pytest.approx(response.cost_usd) and llm.ledger.reserved() == 0


@pytest.mark.parametrize("finish,expected", [("MAX_TOKENS", "max_tokens"), ("SAFETY", "refusal"), ("RECITATION", "refusal")])
def test_how_a_reply_ended_is_put_in_the_projects_own_words(monkeypatch, finish, expected):
    llm, _ = make(monkeypatch, [reply(finish=finish)])
    assert llm.complete(ask(), line="comparison").stop_reason == expected


def test_a_blocked_prompt_counts_as_a_refusal(monkeypatch):
    blocked = reply()
    blocked.candidates = []
    llm, _ = make(monkeypatch, [blocked])
    response = llm.complete(ask(), line="comparison")
    assert response.stop_reason == "refusal" and response.text == ""


def test_a_busy_server_is_tried_again_and_a_bad_request_is_not(monkeypatch):
    llm, sent = make(monkeypatch, [api_error(503), api_error(429), reply()])
    response = llm.complete(ask(), line="comparison")
    assert response.attempts == 3 and len(sent) == 3
    assert llm.ledger.health()["unconfirmed_calls"] == 0  # the API answered with an error, so nothing was billed
    llm, sent = make(monkeypatch, [api_error(400), reply()])
    with pytest.raises(errors.ClientError):
        llm.complete(ask(), line="comparison")
    assert len(sent) == 1 and llm.ledger.spent() == 0 and llm.ledger.reserved() == 0


def test_a_request_that_vanished_stays_counted_at_its_worst_case(monkeypatch):
    llm, sent = make(monkeypatch, [ConnectionError("reset")] * MAX_ATTEMPTS)
    with pytest.raises(ConnectionError):
        llm.complete(ask(), line="comparison")
    health = llm.ledger.health()
    assert len(sent) == MAX_ATTEMPTS and health["unconfirmed_calls"] == MAX_ATTEMPTS and llm.ledger.spent() > 0


def test_a_reply_without_token_counts_is_kept_as_unconfirmed_spend(monkeypatch):
    llm, _ = make(monkeypatch, [reply(usage=False)])
    with pytest.raises(RuntimeError, match="cannot be priced"):
        llm.complete(ask(), line="comparison")
    assert llm.ledger.health()["unconfirmed_calls"] == 1


def test_an_answer_from_another_model_is_rejected(monkeypatch):
    llm, _ = make(monkeypatch, [reply(model="gemini-9-ultra")])
    with pytest.raises(ServedModelMismatch):
        llm.complete(ask(), line="comparison")
    assert llm.ledger.spent() > 0  # it was answered, so it is still counted


def test_each_model_family_gets_its_own_client_and_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GOOGLE_LEDGER_PATH", tmp_path / "ledger-google.sqlite")
    monkeypatch.setattr(config, "LEDGER_PATH", tmp_path / "ledger.sqlite")
    monkeypatch.setattr(llm_module, "load_google_key", lambda: "not-a-real-key")
    monkeypatch.setattr(llm_module, "load_api_key", lambda: "not-a-real-key")
    monkeypatch.setattr(llm_module, "google_cloud_endpoint", lambda: True)
    gemini = client_for("gemini-3.8-flash")
    assert isinstance(gemini, GeminiLLM) and gemini.ledger.path == str(tmp_path / "ledger-google.sqlite")
    assert gemini.ledger.cap_usd == config.GOOGLE_CAP_USD and set(gemini.ledger.lines) == {"adk", "comparison", "changing_docs"}
    claude = client_for("claude-haiku-5-5")
    assert isinstance(claude, AnthropicLLM) and claude.ledger.path == str(tmp_path / "ledger.sqlite")
