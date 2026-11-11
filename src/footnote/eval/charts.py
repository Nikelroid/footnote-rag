"""Draw the results as plain SVG files. No plotting library is needed.

Both charts are the same form: one horizontal bar per row, cut into shares that add
up to 100%. The colours are the first three slots of a palette that was checked for
colour-blind readers, and grey for the outcomes that are neither right nor wrong.
Every share is also printed in the table next to the chart, so nothing depends on
telling two colours apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GREY, PALE = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7", "#e1e0d9"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

BAR, ROW, GAP, ROUND = 20, 36, 2, 4  # bar thickness, row height, gap between segments, rounding of the far end
CHAR = 6.6  # rough width of one character of an 11.5px label, used to decide whether a label fits


def _text(x: float, y: float, text: str, size: float = 12, fill: str = INK, anchor: str = "start", weight: int = 400) -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{escape(text)}</text>'


def _segment(x: float, y: float, width: float, fill: str, last: bool) -> str:
    """One share of a bar. The far end of the bar is rounded, and its start stays square."""
    if not last or width <= ROUND:
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{BAR}" fill="{fill}"/>'
    r = ROUND
    return (
        f'<path d="M{x:.1f},{y:.1f} h{width - r:.1f} a{r},{r} 0 0 1 {r},{r} v{BAR - 2 * r} '
        f'a{r},{r} 0 0 1 -{r},{r} h-{width - r:.1f} z" fill="{fill}"/>'
    )


def stacked_bars(
    title: str, subtitle: str, rows: list[tuple[str, dict[str, float]]], segments: list[tuple[str, str, str, str]], note: str = "", label_width: int = 250, width: int = 820
) -> str:
    """An SVG chart. `rows` are (label, {key: share}). `segments` are (key, legend text, fill, ink of a label inside it)."""
    pad = 24
    left, right = pad + label_width, width - pad - 8
    plot = right - left
    top = pad + 78
    bottom = top + ROW * len(rows)
    height = bottom + 30 + (22 if note else 0) + pad
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="{FONT}" role="img" aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        _text(pad, pad + 16, title, 16, INK, weight=600),
        _text(pad, pad + 36, subtitle, 12.5, SECONDARY),
    ]
    x = pad
    for _, legend, fill, _ in segments:  # the legend names every colour, so identity never rests on colour alone
        out.append(f'<rect x="{x}" y="{pad + 50}" width="11" height="11" rx="2" fill="{fill}"/>')
        out.append(_text(x + 17, pad + 60, legend, 12, SECONDARY))
        x += 17 + len(legend) * 6.9 + 22
    for share in (0, 0.25, 0.5, 0.75, 1.0):
        gx = left + plot * share
        out.append(f'<line x1="{gx:.1f}" y1="{top - 4}" x2="{gx:.1f}" y2="{bottom}" stroke="{AXIS if share == 0 else GRID}" stroke-width="1"/>')
        out.append(_text(gx, bottom + 16, f"{round(100 * share)}%", 11, MUTED, "middle"))
    for i, (label, shares) in enumerate(rows):
        y = top + i * ROW + (ROW - BAR) / 2
        out.append(_text(left - 12, y + BAR / 2 + 4, label, 12.5, INK, "end"))
        shown = [(key, fill, ink, shares.get(key, 0.0)) for key, _, fill, ink in segments if shares.get(key, 0.0) > 0]
        x = float(left)
        for j, (key, fill, ink, share) in enumerate(shown):
            last = j == len(shown) - 1
            full = plot * share
            drawn = full if last else max(full - GAP, 0.0)  # a gap in the surface colour separates two shares
            out.append(_segment(x, y, drawn, fill, last))
            label_text = f"{100 * share:.1f}%"
            if drawn >= len(label_text) * CHAR + 12:  # a label that does not fit is left to the table
                out.append(_text(x + drawn / 2, y + BAR / 2 + 4, label_text, 11.5, ink, "middle", 600))
            x += full
    if note:
        out.append(_text(pad, bottom + 42, note, 11.5, MUTED))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def bars(title: str, subtitle: str, rows: list[tuple[str, float]], note: str = "", label_width: int = 250, width: int = 820) -> str:
    """An SVG chart of one share per row. One series, so one colour and no legend. Every bar carries its value."""
    pad = 24
    left, right = pad + label_width, width - pad - 8
    plot = right - left
    top = pad + 56
    bottom = top + ROW * len(rows)
    height = bottom + 30 + (22 if note else 0) + pad
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="{FONT}" role="img" aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        _text(pad, pad + 16, title, 16, INK, weight=600),
        _text(pad, pad + 36, subtitle, 12.5, SECONDARY),
    ]
    for share in (0, 0.25, 0.5, 0.75, 1.0):
        gx = left + plot * share
        out.append(f'<line x1="{gx:.1f}" y1="{top - 4}" x2="{gx:.1f}" y2="{bottom}" stroke="{AXIS if share == 0 else GRID}" stroke-width="1"/>')
        out.append(_text(gx, bottom + 16, f"{round(100 * share)}%", 11, MUTED, "middle"))
    for i, (label, share) in enumerate(rows):
        y = top + i * ROW + (ROW - BAR) / 2
        out.append(_text(left - 12, y + BAR / 2 + 4, label, 12.5, INK, "end"))
        drawn = plot * share
        out.append(_segment(left, y, drawn, BLUE, True))
        value = f"{100 * share:.1f}%"
        if left + drawn + 10 + len(value) * CHAR <= right:  # the value sits past the end of the bar when there is room
            out.append(_text(left + drawn + 8, y + BAR / 2 + 4, value, 11.5, INK, "start", 600))
        else:
            out.append(_text(left + drawn - 8, y + BAR / 2 + 4, value, 11.5, "#ffffff", "end", 600))
    if note:
        out.append(_text(pad, bottom + 42, note, 11.5, MUTED))
    out.append("</svg>")
    return "\n".join(out) + "\n"


RETRIEVAL_ROWS = [
    ("bm25", "Keyword search (BM25)"),
    ("vector", "Vector search"),
    ("hybrid+rerank", "Both, reranked"),
    ("hybrid+rerank, routed", "and routed to the filing"),
    ("hybrid+rerank, routed, statements first", "and main statements first"),
]


def retrieval_chart(result: dict, pages: int = 8) -> str:
    """How often the evidence page is among the pages the model reads, from the JSON that `footnote retrieval-check` writes."""
    rows = [(label, result["rows"][name][f"hit@{pages}"]) for name, label in RETRIEVAL_ROWS if name in result["rows"]]
    subtitle = f"{result['questions']} FinanceBench questions ({result['split']}). Each row adds one step to the row above it."
    return bars(f"How often the evidence page is among the {pages} pages the model reads", subtitle, rows, "Search only. No model is called for this measurement.")


def ladder_chart(report: dict, subtitle: str, note: str = "") -> str:
    """Correct, wrong and declined answers for each system, from the JSON that `footnote report` writes."""
    names = {
        "floor": "No documents",
        "rag": "Search once, then answer",
        "agent": "Agent: check, search again",
        "ceiling": "Evidence page handed over",
    }
    rows = [
        (names.get(s["system"], s["system"]), {"correct": s["correct"]["mean"], "incorrect": s["incorrect"]["mean"], "refused": s["refused"]["mean"]})
        for s in report["runs"]
    ]
    segments = [("correct", "Correct", BLUE, "#ffffff"), ("incorrect", "Wrong", ORANGE, "#ffffff"), ("refused", "Declines to answer", GREY, INK)]
    return stacked_bars("What the assistant answers, by how it gets its pages", subtitle, rows, segments, note, label_width=200)


STALE_ROWS = [
    ("synced", "Old page replaced (synced)"),
    ("appended-old-first", "Both versions, old page first"),
    ("appended-new-first", "Both versions, new page first"),
    ("dated-old-first", "Both, with filing dates, old first"),
    ("dated-new-first", "Both, with filing dates, new first"),
    ("dated-rule-old-first", "Dates and a rule, old first"),
    ("dated-rule-new-first", "Dates and a rule, new first"),
]

STALE_SUBTITLES = {
    "moved": "An amount was moved between two lines, so the page still adds up.",
    "sum": "One amount was changed, so a total on the page no longer adds up.",
    "free": "One amount was changed on a line that no total on the page depends on.",
    "benchmark": "The amount a benchmark question asks for was changed.",
    "wrong": "The page comes from another company and has a line with the same name.",
    "injected": "A sentence addressed to AI assistants was added to the right page.",
    "index": "The amended filing is in the real index, and the question goes through search.",
}

# For the two tests that are not about an amendment: the title, the rows, and what each colour stands for.
# Blue is always the answer one would want, and orange the answer that misleads.
AMENDED_SEGMENTS = [
    ("new", "New amount", BLUE, "#ffffff"),
    ("old", "Old amount", ORANGE, "#ffffff"),
    ("both", "Gives both amounts", AQUA, "#ffffff"),
    ("declined", "Declines", GREY, INK),
    ("other", "Other", PALE, INK),
]

OTHER_TESTS = {
    "index": (
        "After a filing changes, which amount does the whole pipeline give?",
        [("synced", "Old filing replaced (synced)"), ("appended", "Amended filing added, old one kept"), ("fixed", "Both kept, with dates and the rule")],
        AMENDED_SEGMENTS,
    ),
    "wrong": (
        "Handed another company's page, what does the assistant answer?",
        [("honest", "Labelled as the other company's filing"), ("mislabelled", "Labelled as the filing asked about")],
        [
            ("declined", "Says it is not there", BLUE, "#ffffff"),
            ("new", "Other company's amount", ORANGE, "#ffffff"),
            ("old", "Right amount, from memory", AQUA, "#ffffff"),
            ("both", "Gives both", GREY, INK),
            ("other", "Other", PALE, INK),
        ],
    ),
    "injected": (
        "A page tells the assistant to report another amount. Does it?",
        [("plain", "Page with an injected instruction"), ("defended", "Same page, instructions warn against it")],
        [
            ("old", "Amount printed on the page", BLUE, "#ffffff"),
            ("new", "Injected amount", ORANGE, "#ffffff"),
            ("both", "Gives both amounts", AQUA, "#ffffff"),
            ("declined", "Declines", GREY, INK),
            ("other", "Other", PALE, INK),
        ],
    ),
}


def stale_chart(summary: dict, group: str, note: str = "") -> str:
    """Which amount came back in each setting of one test, from the JSON that `footnote stale-report` writes."""
    title, names, segments = OTHER_TESTS.get(
        group,
        ("After a filing changes, which amount does the assistant give?", STALE_ROWS, AMENDED_SEGMENTS),
    )
    settings = summary["groups"][group]["settings"]
    rows = [(label, {key: settings[name][key]["mean"] for key in ("new", "old", "both", "declined", "other")}) for name, label in names if name in settings]
    subtitle = f"{STALE_SUBTITLES[group]} {summary['model']} is then asked for that line."
    return stacked_bars(title, subtitle, rows, segments, note)


MODEL_NAMES = {"claude-haiku-5-5": "Claude Haiku 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5", "gemini-3.8-flash": "Gemini 3.8 Flash"}
MODEL_COLOURS = (BLUE, ORANGE, AQUA)  # one colour a model, in the palette's fixed order
THIN = 10  # bar thickness where each row holds one bar per model


def grouped_bars(
    title: str, subtitle: str, rows: list[tuple[str, dict[str, float | None]]], series: list[tuple[str, str]], note: str = "", label_width: int = 290, width: int = 820
) -> str:
    """An SVG chart with one bar per series in every row. `series` are (key, legend text). A missing value draws nothing."""
    pad = 24
    left, right = pad + label_width, width - pad - 8
    plot = right - left
    row = len(series) * (THIN + GAP) + 14
    top = pad + 78
    bottom = top + row * len(rows)
    height = bottom + 30 + (22 if note else 0) + pad
    colours = dict(zip((key for key, _ in series), MODEL_COLOURS))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="{FONT}" role="img" aria-label="{escape(title)}">',
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        _text(pad, pad + 16, title, 16, INK, weight=600),
        _text(pad, pad + 36, subtitle, 12.5, SECONDARY),
    ]
    x = pad
    for key, legend in series:
        out.append(f'<rect x="{x}" y="{pad + 50}" width="11" height="11" rx="2" fill="{colours[key]}"/>')
        out.append(_text(x + 17, pad + 60, legend, 12, SECONDARY))
        x += 17 + len(legend) * 6.9 + 22
    for share in (0, 0.25, 0.5, 0.75, 1.0):
        gx = left + plot * share
        out.append(f'<line x1="{gx:.1f}" y1="{top - 4}" x2="{gx:.1f}" y2="{bottom}" stroke="{AXIS if share == 0 else GRID}" stroke-width="1"/>')
        out.append(_text(gx, bottom + 16, f"{round(100 * share)}%", 11, MUTED, "middle"))
    for i, (label, values) in enumerate(rows):
        y = top + i * row + 7
        out.append(_text(left - 12, y + (row - 14) / 2 + 4, label, 12.5, INK, "end"))
        for j, (key, _) in enumerate(series):
            share = values.get(key)
            if share is None:
                continue
            by = y + j * (THIN + GAP)
            drawn = plot * share
            if drawn > 0:
                r = min(ROUND, drawn)
                out.append(
                    f'<path d="M{left:.1f},{by:.1f} h{drawn - r:.1f} a{r},{r} 0 0 1 {r},{r} v{THIN - 2 * r} a{r},{r} 0 0 1 -{r},{r} h-{drawn - r:.1f} z" fill="{colours[key]}"/>'
                )
            value = f"{100 * share:.0f}%"
            inside = left + drawn + 8 + len(value) * CHAR > right
            out.append(_text(left + drawn - 6 if inside else left + drawn + 6, by + THIN - 1, value, 10.5, "#ffffff" if inside else SECONDARY, "end" if inside else "start", 600))
    if note:
        out.append(_text(pad, bottom + 42, note, 11.5, MUTED))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def scorecard_chart(results_dir: Path | str, models: tuple[str, ...] = tuple(MODEL_NAMES)) -> str | None:
    """How often each model gives a misleading amount in each test, from the result files that are there. None when there are none."""
    from ..rag.stale import scorecard

    found, rows = scorecard(results_dir, models)
    if not found:
        return None
    return grouped_bars(
        "How often does the answer give the misleading amount?",
        "Real statement pages, one line asked for. The misleading amount is the old one, another company's, or an injected one.",
        rows,
        [(model, MODEL_NAMES.get(model, model)) for model in found],
        "Each bar is the share of answers that give the misleading amount, alone or first. Shorter is better.",
    )


def models_chart(reports: dict[str, dict]) -> str:
    """Correct answers for the same systems on different answering models, from the JSON that `footnote report` writes for each."""
    names = [("floor", "No documents"), ("agent", "Agent with search"), ("ceiling", "Evidence page handed over")]
    by_model = {model: {s["system"]: s["correct"]["mean"] for s in report["runs"]} for model, report in reports.items()}
    rows = [(label, {model: scores.get(system) for model, scores in by_model.items()}) for system, label in names]
    first = next(iter(reports.values()))["runs"][0]
    return grouped_bars(
        "Correct answers, by the model that answers",
        f"{first['n_cases']} FinanceBench questions. The search, the prompts and the judge are the same for every model.",
        rows,
        [(model, MODEL_NAMES.get(model, model)) for model in reports],
        "Longer is better.",
        label_width=210,
    )


def write(svg: str, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg)
    return path


def load(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())
