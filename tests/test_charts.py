import xml.etree.ElementTree as ET

import pytest

from footnote.eval.charts import BAR, GAP, bars, grouped_bars, ladder_chart, models_chart, retrieval_chart, scorecard_chart, stacked_bars, stale_chart

SVG = "{http://www.w3.org/2000/svg}"
SEGMENTS = [("a", "First", "#2a78d6", "#ffffff"), ("b", "Second", "#eb6834", "#ffffff"), ("c", "Third", "#c3c2b7", "#0b0b0b")]


def marks(svg: str):
    root = ET.fromstring(svg)  # the file has to be well-formed, or no browser will draw it
    bars = [el for el in root if el.tag in (SVG + "rect", SVG + "path") and el.get("height") in (str(BAR), None) and el.get("fill") in {s[2] for s in SEGMENTS}]
    return root, bars


def test_shares_become_widths_with_a_gap_between_them():
    svg = stacked_bars("Title", "Subtitle", [("Row", {"a": 0.5, "b": 0.5})], SEGMENTS, label_width=200, width=800)
    root, _ = marks(svg)
    plot = 800 - 24 - 8 - (24 + 200)
    first = next(el for el in root.iter(SVG + "rect") if el.get("height") == str(BAR))
    assert float(first.get("width")) == plot / 2 - GAP  # the first share stops short, and the surface shows through
    last = next(el for el in root.iter(SVG + "path"))
    assert last.get("fill") == "#eb6834"  # the far end of the bar is the rounded one
    texts = [el.text for el in root.iter(SVG + "text")]
    assert texts.count("50.0%") == 2 and "First" in texts and "Third" in texts  # the legend names every colour, used or not


def test_a_label_that_does_not_fit_is_left_out_and_an_empty_share_draws_nothing():
    svg = stacked_bars("Title", "Subtitle", [("Row", {"a": 0.97, "b": 0.03, "c": 0.0})], SEGMENTS)
    root, _ = marks(svg)
    texts = [el.text for el in root.iter(SVG + "text")]
    assert "97.0%" in texts and "3.0%" not in texts
    fills = [el.get("fill") for el in root.iter() if el.get("height") == str(BAR) or el.tag == SVG + "path"]
    assert "#c3c2b7" not in fills[:2] and len([f for f in fills if f in ("#2a78d6", "#eb6834")]) == 2


def test_the_two_charts_read_the_result_files():
    stat = lambda m: {"mean": m, "lo": 0.0, "hi": 1.0, "n_cases": 10}
    report = {"runs": [{"system": "rag", "correct": stat(0.8), "incorrect": stat(0.15), "refused": stat(0.05)}]}
    ladder = ladder_chart(report, "150 questions")
    assert "Search once, then answer" in ladder and "80.0%" in ladder
    policy = {key: stat(value) for key, value in (("new", 0.6), ("old", 0.3), ("both", 0.1), ("declined", 0.0), ("other", 0.0))}
    summary = {"model": "claude-haiku-5-5", "groups": {"moved": {"settings": {"appended-old-first": policy, "unknown": policy}}}}
    stale = stale_chart(summary, "moved")
    assert "Both versions, old page first" in stale and "30.0%" in stale and "unknown" not in stale
    assert "the page still adds up" in stale  # the subtitle says which kind of amendment this is
    ET.fromstring(ladder), ET.fromstring(stale)


def test_one_bar_per_row_with_its_value_past_the_end_or_inside_when_there_is_no_room():
    svg = bars("Title", "Subtitle", [("Short", 0.2), ("Full", 1.0)], label_width=200, width=800)
    root = ET.fromstring(svg)
    plot = 800 - 24 - 8 - (24 + 200)
    values = {el.text: el for el in root.iter(SVG + "text") if el.text in ("20.0%", "100.0%")}
    assert float(values["20.0%"].get("x")) == pytest.approx(24 + 200 + plot * 0.2 + 8) and values["20.0%"].get("text-anchor") == "start"
    assert values["100.0%"].get("text-anchor") == "end" and values["100.0%"].get("fill") == "#ffffff"  # no room past a full bar
    assert len(list(root.iter(SVG + "path"))) == 2 and not any(el.get("rx") for el in root.iter(SVG + "rect"))  # one colour, so no legend


def test_the_retrieval_chart_shows_the_steps_in_order():
    rows = {name: {"hit@8": share} for name, share in (("bm25", 0.18), ("vector", 0.57), ("hybrid+rerank", 0.56), ("hybrid+rerank, routed", 0.63), ("hybrid+rerank, routed, statements first", 0.77), ("hybrid", 0.42))}
    svg = retrieval_chart({"questions": 99, "split": "test", "rows": rows})
    order = [svg.index(text) for text in ("Keyword search (BM25)", "Vector search", "Both, reranked", "and routed to the filing", "and main statements first")]
    assert order == sorted(order) and "77.0%" in svg and "42.0%" not in svg  # the plain hybrid row is not one of the steps
    ET.fromstring(svg)


def test_the_other_two_tests_get_their_own_rows_and_legend():
    stat = lambda m: {"mean": m, "lo": 0.0, "hi": 1.0, "n_cases": 10}
    outcome = {key: stat(value) for key, value in (("new", 0.1), ("old", 0.0), ("both", 0.0), ("declined", 0.9), ("other", 0.0))}
    wrong = stale_chart({"model": "m", "groups": {"wrong": {"settings": {"honest": outcome, "mislabelled": outcome}}}}, "wrong")
    assert "Says it is not there" in wrong and "Labelled as the filing asked about" in wrong and "90.0%" in wrong
    injected = stale_chart({"model": "m", "groups": {"injected": {"settings": {"plain": outcome}}}}, "injected")
    assert "Injected amount" in injected and "Page with an injected instruction" in injected and "instructions warn against it" not in injected
    ET.fromstring(wrong), ET.fromstring(injected)


def test_grouped_bars_draw_one_bar_per_model_and_skip_what_is_missing():
    svg = grouped_bars("Title", "Subtitle", [("Row", {"a": 0.5, "b": 0.0, "c": None})], [("a", "Model A"), ("b", "Model B"), ("c", "Model C")])
    root = ET.fromstring(svg)
    texts = [el.text for el in root.iter(SVG + "text")]
    assert "50%" in texts and "0%" in texts and texts.count("Model C") == 1  # a zero is still written out, a missing value is not
    assert len(list(root.iter(SVG + "path"))) == 1  # only the bar with a length is drawn


def test_the_scorecard_reads_whatever_result_files_exist(tmp_path):
    import json

    assert scorecard_chart(tmp_path) is None
    stat = lambda m: {"mean": m, "lo": 0.0, "hi": 1.0, "n_cases": 10}
    moved = {"groups": {"moved": {"settings": {"synced": {"leads_old": stat(0.0)}, "appended-old-first": {"leads_old": stat(0.87)}}}}}
    (tmp_path / "amended-moved-claude-haiku-5-5.json").write_text(json.dumps(moved))
    svg = scorecard_chart(tmp_path)
    assert "Claude Haiku 5.5" in svg and "Claude Sonnet 5.5" not in svg and "87%" in svg and "Both versions kept, old page first" in svg
    ET.fromstring(svg)
    from footnote.rag.stale import render_scorecard, scorecard

    found, rows = scorecard(tmp_path, ("claude-haiku-5-5", "claude-sonnet-5-5"))
    table = render_scorecard(found, rows)
    assert found == ["claude-haiku-5-5"] and "| Both versions kept, old page first | 87.0% |" in table and "| Page with an injected instruction | not run |" in table


def test_the_models_chart_puts_the_same_systems_side_by_side():
    stat = lambda m: {"mean": m, "lo": 0.0, "hi": 1.0, "n_cases": 150}
    def report(model, floor, agent):
        return {"runs": [{"system": "floor", "model": model, "n_cases": 150, "correct": stat(floor)}, {"system": "agent", "model": model, "n_cases": 150, "correct": stat(agent)}]}
    svg = models_chart({"claude-haiku-5-5": report("claude-haiku-5-5", 0.34, 0.86), "gemini-3.8-flash": report("gemini-3.8-flash", 0.2, 0.8)})
    assert "Gemini 3.8 Flash" in svg and "86%" in svg and "20%" in svg and "Agent with search" in svg
    ET.fromstring(svg)
