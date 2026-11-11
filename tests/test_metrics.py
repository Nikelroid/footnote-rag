import numpy as np
import pytest
from scipy import stats

from footnote.eval.metrics import (
    cluster_bootstrap,
    cohen_kappa,
    confusion,
    flip_rate,
    mcnemar_exact,
    paired_difference,
    retrieval_scores,
    wilson_interval,
)


@pytest.mark.parametrize("k,n", [(0, 150), (14, 150), (75, 150), (128, 150), (150, 150), (3, 7)])
def test_wilson_matches_scipy(k, n):
    expected = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    lo, hi = wilson_interval(k, n)
    assert lo == pytest.approx(expected.low, abs=1e-9) and hi == pytest.approx(expected.high, abs=1e-9)


@pytest.mark.parametrize("b,c", [(0, 0), (5, 5), (12, 3), (1, 9), (30, 10)])
def test_mcnemar_matches_exact_binomial(b, c):
    expected = 1.0 if b + c == 0 else stats.binomtest(min(b, c), b + c, 0.5).pvalue
    assert mcnemar_exact(b, c) == pytest.approx(expected, abs=1e-12)


def test_cluster_bootstrap_keeps_repeats_together():
    mixed = {i: [1.0, 0.0] for i in range(50)}  # every question is a coin flip between runs
    r = cluster_bootstrap(mixed)
    lo, hi = wilson_interval(25, 50)
    # No spread between questions, so resampling shows none. The interval falls back to the wider Wilson one.
    assert r["mean"] == 0.5 and (r["lo"], r["hi"]) == (lo, hi)


def test_an_all_or_nothing_result_does_not_claim_certainty():
    nothing = cluster_bootstrap({i: [0.0, 0.0, 0.0] for i in range(150)})
    assert nothing["mean"] == 0.0 and nothing["lo"] == 0.0 and 0.02 < nothing["hi"] < 0.03
    everything = cluster_bootstrap({i: [1.0] for i in range(20)})
    assert everything["hi"] == 1.0 and 0.8 < everything["lo"] < 0.9


def test_cluster_bootstrap_covers_a_binomial_proportion():
    values = {i: [1.0 if i < 60 else 0.0] for i in range(150)}
    r = cluster_bootstrap(values)
    lo, hi = wilson_interval(60, 150)
    assert r["mean"] == pytest.approx(0.4)
    assert r["lo"] == pytest.approx(lo, abs=0.02) and r["hi"] == pytest.approx(hi, abs=0.02)


def test_paired_difference_sees_a_real_gain_and_ignores_noise():
    a = {i: [0.0] for i in range(100)}
    b = {i: [1.0 if i < 30 else 0.0] for i in range(100)}
    gain = paired_difference(a, b)
    assert gain["diff"] == pytest.approx(0.30) and gain["lo"] > 0.15 and gain["p_value"] < 0.001 and gain["b_better"] == 30
    rng = np.random.default_rng(1)
    x = {i: [float(rng.integers(0, 2))] for i in range(100)}
    y = {i: [float(rng.integers(0, 2))] for i in range(100)}
    noise = paired_difference(x, y)
    assert noise["lo"] < 0 < noise["hi"] and noise["p_value"] > 0.05


def test_paired_difference_uses_only_shared_questions():
    assert paired_difference({1: [1.0], 2: [0.0]}, {2: [1.0], 3: [1.0]})["n_cases"] == 1


def test_flip_rate():
    assert flip_rate({1: [1, 1, 1], 2: [1, 0, 1], 3: [0, 0, 0], 4: [0, 1, 1]}) == 0.5


def test_kappa_on_a_known_table():
    table = confusion(["a"] * 25 + ["b"] * 25, ["a"] * 20 + ["b"] * 5 + ["a"] * 10 + ["b"] * 15, ["a", "b"])
    assert table == [[20, 5], [10, 15]]
    assert cohen_kappa(table) == pytest.approx(0.4)
    assert cohen_kappa([[10, 0], [0, 10]]) == 1.0


def test_retrieval_scores():
    gold = {("doc", 59), ("doc", 60)}
    ranked = [("other", 1), ("doc", 12), ("doc", 60), ("doc", 59)]
    s = retrieval_scores(gold, ranked, ks=(1, 3, 5))
    assert s["hit@1"] == 0 and s["doc_hit@1"] == 0 and s["doc_hit@3"] == 1
    assert s["hit@3"] == 1 and s["recall@3"] == 0.5 and s["recall@5"] == 1.0
    assert s["mrr"] == pytest.approx(1 / 3)
    assert retrieval_scores(gold, [])["mrr"] == 0.0


def test_retrieval_scores_count_a_page_once_however_many_chunks_it_has():
    gold = {("doc", 60)}
    many_chunks = [("doc", 12)] * 5 + [("doc", 60)]
    s = retrieval_scores(gold, many_chunks, ks=(1, 5))
    assert s["hit@5"] == 1.0 and s["mrr"] == 0.5  # page 60 is the second page, not the sixth item
    assert s == retrieval_scores(gold, [("doc", 12), ("doc", 60)], ks=(1, 5))
