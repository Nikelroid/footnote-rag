"""Check the statistics on made-up benchmarks where the truth is known.

A 95% interval should contain the true value about 95 times in 100, and a test
at the 5% level should raise a false alarm about 5 times in 100. Neither claim
can be checked on the real benchmark, where the truth is unknown. Here the
questions are simulated, so both can be counted. The same simulation says how
large a real gain has to be before this benchmark can see it.

The made-up benchmark has three kinds of question. Some are always answered
correctly, some never, and the rest succeed on some runs and fail on others.
"""

from __future__ import annotations

import numpy as np

from .metrics import cluster_bootstrap, paired_difference

SHARE_ALWAYS = 0.45
SHARE_NEVER = 0.35
UNSTABLE_RANGE = (0.2, 0.8)
TRUE_ACCURACY = SHARE_ALWAYS + (1 - SHARE_ALWAYS - SHARE_NEVER) * sum(UNSTABLE_RANGE) / 2


def success_probabilities(rng: np.random.Generator, n: int) -> np.ndarray:
    kind = rng.random(n)
    unstable = rng.uniform(*UNSTABLE_RANGE, size=n)
    return np.where(kind < SHARE_ALWAYS, 1.0, np.where(kind < SHARE_ALWAYS + SHARE_NEVER, 0.0, unstable))


def runs(rng: np.random.Generator, p: np.ndarray, reps: int) -> dict[int, list[float]]:
    outcomes = (rng.random((len(p), reps)) < p[:, None]).astype(float)
    return {i: list(row) for i, row in enumerate(outcomes)}


def interval_coverage(n_sims: int = 2000, n: int = 150, reps: int = 3, seed: int = 0, n_boot: int = 2000) -> float:
    """Share of simulated benchmarks whose 95% interval contains the true accuracy."""
    rng = np.random.default_rng(seed)
    hits = 0
    for sim in range(n_sims):
        result = cluster_bootstrap(runs(rng, success_probabilities(rng, n), reps), n_boot=n_boot, seed=sim)
        hits += result["lo"] <= TRUE_ACCURACY <= result["hi"]
    return hits / n_sims


def detection_rate(
    gain: float, n_sims: int = 1000, n: int = 150, reps: int = 3, seed: int = 1, alpha: float = 0.05, n_boot: int = 1000, n_perm: int = 5000
) -> float:
    """Share of simulated comparisons that the paired test calls significant.

    With gain 0 the two systems are equally good and every call is a false alarm.
    With a positive gain, system B also answers that share of the questions which
    system A never gets right, and the result is the power of the test.
    """
    rng = np.random.default_rng(seed)
    calls = 0
    for sim in range(n_sims):
        p_a = success_probabilities(rng, n)
        p_b = p_a.copy()
        never = np.flatnonzero(p_a == 0.0)
        fixed = rng.choice(never, size=min(len(never), round(gain * n)), replace=False)
        p_b[fixed] = 1.0
        result = paired_difference(runs(rng, p_a, reps), runs(rng, p_b, reps), n_boot=n_boot, n_perm=n_perm, seed=sim)
        calls += result["p_value"] < alpha
    return calls / n_sims


def check(n_sims: int = 2000, n: int = 150, reps: int = 3) -> dict:
    return {
        "questions": n,
        "runs_per_question": reps,
        "simulations": n_sims,
        "interval_coverage": interval_coverage(n_sims, n, reps),
        "false_alarm_rate": detection_rate(0.0, n_sims, n, reps),
        "power_at_gain": {f"{g:.2f}": detection_rate(g, n_sims // 2, n, reps) for g in (0.03, 0.05, 0.08, 0.10)},
    }
