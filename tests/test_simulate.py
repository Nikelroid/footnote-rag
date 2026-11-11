from footnote.eval import simulate


def test_simulation_runs_and_is_repeatable():
    first = simulate.check(n_sims=40, n=60, reps=2)
    assert first == simulate.check(n_sims=40, n=60, reps=2)
    assert 0.8 <= first["interval_coverage"] <= 1.0 and first["false_alarm_rate"] <= 0.2
    assert first["power_at_gain"]["0.10"] >= first["power_at_gain"]["0.03"]


def test_assumed_accuracy_matches_the_mix_of_question_kinds():
    assert simulate.TRUE_ACCURACY == 0.45 + 0.20 * 0.5
