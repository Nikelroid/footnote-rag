import pytest

from footnote import config
from footnote.pricing import UnknownModel, Usage, cost_usd, resolve_model, worst_case_cost_usd


def test_haiku_base_rates():
    assert cost_usd("claude-haiku-5-5", Usage(input_tokens=100_000, output_tokens=10_000)) == pytest.approx(0.01 + 0.005)


def test_haiku_long_context_threshold_counts_cached_tokens():
    at_limit = Usage(input_tokens=60_000, cache_read_tokens=40_000, output_tokens=1_000)
    over = Usage(input_tokens=60_000, cache_read_tokens=40_001, output_tokens=1_000)
    assert cost_usd("claude-haiku-5-5", at_limit) == pytest.approx((60_000 * 0.10 + 40_000 * 0.01 + 1_000 * 0.50) / 1e6)
    assert cost_usd("claude-haiku-5-5", over) == pytest.approx((60_000 * 0.50 + 40_001 * 0.05 + 1_000 * 2.50) / 1e6)


def test_cache_writes_and_batch_discount():
    usage = Usage(input_tokens=1_000, output_tokens=1_000, cache_write_5m_tokens=1_000, cache_write_1h_tokens=1_000, cache_read_tokens=1_000)
    full = (1_000 * 2.00 + 1_000 * 10.00 + 1_000 * 2.50 + 1_000 * 4.00 + 1_000 * 0.10) / 1e6
    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(full)
    assert cost_usd("claude-sonnet-5-5", usage, batch=True) == pytest.approx(full / 2)


def test_sonnet_has_no_long_context_premium():
    big = Usage(input_tokens=500_000, output_tokens=0)
    assert cost_usd("claude-sonnet-5-5", big) == pytest.approx(1.0)


def test_snapshot_ids_resolve_and_unknown_models_fail_loudly():
    assert resolve_model("claude-haiku-5-5-20261001") == "claude-haiku-5-5"
    with pytest.raises(UnknownModel):
        cost_usd("claude-haiku-9", Usage(input_tokens=1))


def test_worst_case_is_an_upper_bound_on_a_typical_request():
    chars, max_tokens = 9_000, 4096
    typical = cost_usd("claude-haiku-5-5", Usage(input_tokens=chars // 3, output_tokens=600))
    assert worst_case_cost_usd("claude-haiku-5-5", chars, max_tokens) > typical


def test_budget_lines_add_up_to_the_hard_cap():
    assert sum(config.BUDGET_LINES.values()) == pytest.approx(config.HARD_CAP_USD)
    assert config.HARD_CAP_USD == 90.0  # the ceiling for the project is $100. The cap leaves room under it.
    assert config.GOOGLE_CAP_USD < 39.0  # the Google Cloud credit
