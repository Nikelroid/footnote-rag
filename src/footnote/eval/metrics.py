"""Statistics for the results table.

Every question is answered by every system, so systems are compared on the same
questions (a paired design). Answers also vary between runs, so each question is
run several times. The functions here treat the question as the unit that is
resampled, and keep its repeats together.
"""

from __future__ import annotations

import math
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np

Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% interval for a proportion from one run. Better than the normal interval near 0 and 1."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def per_case_means(values: Mapping[Hashable, Sequence[float]]) -> dict[Hashable, float]:
    return {case: float(np.mean(v)) for case, v in values.items() if len(v) > 0}


def cluster_bootstrap(
    values: Mapping[Hashable, Sequence[float]], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Mean over questions with a percentile interval, resampling whole questions."""
    means = np.array(list(per_case_means(values).values()), dtype=float)
    n = len(means)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_cases": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = means[idx].mean(axis=1)
    lo, hi = (float(v) for v in np.quantile(boots, [alpha / 2, 1 - alpha / 2]))
    mean = float(means.mean())
    if hi - lo < 1e-12:
        # Every question gave the same result, so resampling shows no spread. That is not
        # certainty: 0 right out of 150 still allows a true rate of a few percent.
        w_lo, w_hi = wilson_interval(round(mean * n), n)
        lo, hi = min(lo, w_lo), max(hi, w_hi)
    return {"mean": mean, "lo": lo, "hi": hi, "n_cases": n}


def paired_difference(
    a: Mapping[Hashable, Sequence[float]],
    b: Mapping[Hashable, Sequence[float]],
    n_boot: int = 10_000,
    n_perm: int = 20_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """B minus A on the questions both systems answered.

    The interval comes from resampling questions. The p-value comes from a
    sign-flip permutation test on the per-question differences, which assumes
    nothing about their distribution.
    """
    ma, mb = per_case_means(a), per_case_means(b)
    cases = sorted(set(ma) & set(mb), key=str)
    d = np.array([mb[c] - ma[c] for c in cases], dtype=float)
    n = len(d)
    if n == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_value": float("nan"), "n_cases": 0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    observed = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm = np.abs((signs * d).mean(axis=1))
    p = (np.sum(perm >= observed - 1e-12) + 1) / (n_perm + 1)
    return {
        "diff": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_value": float(p),
        "n_cases": n,
        "b_better": int(np.sum(d > 0)),
        "a_better": int(np.sum(d < 0)),
    }


def mcnemar_exact(b_only: int, a_only: int) -> float:
    """Two-sided exact McNemar test for one run of each system. Only the discordant pairs matter."""
    n = b_only + a_only
    if n == 0:
        return 1.0
    k = min(b_only, a_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def flip_rate(values: Mapping[Hashable, Sequence[float]]) -> float:
    """Share of questions whose repeats do not all agree. This is the run-to-run noise."""
    repeated = [v for v in values.values() if len(v) > 1]
    if not repeated:
        return float("nan")
    return sum(1 for v in repeated if len(set(v)) > 1) / len(repeated)


def confusion(truth: Iterable[str], predicted: Iterable[str], labels: Sequence[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    table = [[0] * len(labels) for _ in labels]
    for t, p in zip(truth, predicted, strict=True):
        table[index[t]][index[p]] += 1
    return table


def cohen_kappa(table: Sequence[Sequence[int]]) -> float:
    """Agreement between two graders beyond what chance would give."""
    m = np.array(table, dtype=float)
    total = m.sum()
    if total == 0:
        return float("nan")
    observed = np.trace(m) / total
    expected = float((m.sum(axis=0) * m.sum(axis=1)).sum()) / (total * total)
    if expected == 1.0:
        return 1.0
    return float((observed - expected) / (1 - expected))


def retrieval_scores(
    gold: set[tuple[str, int]], ranked: Sequence[tuple[str, int]], ks: Sequence[int] = (1, 3, 5, 10, 20)
) -> dict[str, float]:
    """How well a ranked list of (document, page) hits covers the evidence pages.

    hit@k asks whether any evidence page is in the top k, recall@k how many of them,
    and doc_hit@k whether the right filing shows up at all.
    """
    # Several chunks can come from one page. Pages are counted once, at their best
    # rank, so the score does not depend on how the text was cut into chunks.
    ranked = list(dict.fromkeys(ranked))
    out: dict[str, float] = {}
    gold_docs = {doc for doc, _ in gold}
    for k in ks:
        top = list(ranked[:k])
        found = gold & set(top)
        out[f"hit@{k}"] = 1.0 if found else 0.0
        out[f"recall@{k}"] = len(found) / len(gold) if gold else float("nan")
        out[f"doc_hit@{k}"] = 1.0 if gold_docs & {doc for doc, _ in top} else 0.0
    rank = next((i + 1 for i, hit in enumerate(ranked) if hit in gold), None)
    out["mrr"] = 1.0 / rank if rank else 0.0
    return out
