"""Command line entry point. Run `footnote --help` for the list of commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config


def _questions(split: str, limit: int | None):
    from .data.financebench import load_questions

    questions = [q for q in load_questions() if split == "all" or q.split == split]
    return questions[:limit] if limit else questions


def cmd_data(args) -> int:
    from .data import download

    manifest = download.load_manifest()
    entries = download.core_entries(manifest)
    if args.pdfs == "all":
        entries += download.pdf_entries(manifest)
    elif args.pdfs == "needed":
        from .data.financebench import load_questions

        if args.action == "fetch":
            # Which filings are needed is read from the question file, so the benchmark files are fetched before the PDFs.
            print(f"Fetching the {len(entries)} benchmark files first, {sum(e['size'] for e in entries) / 1e6:.1f} MB")
            download.fetch(manifest, entries)
        entries += download.pdf_entries(manifest, {q.doc_name for q in load_questions()})
    total = sum(e["size"] for e in entries)
    if args.action == "verify":
        bad = download.verify(manifest, entries)
        print(f"{len(entries) - len(bad)} of {len(entries)} files match the manifest")
        for path in bad:
            print(f"  missing or changed: {path}")
        return 1 if bad else 0
    print(f"Fetching {len(entries)} files, {total / 1e6:.1f} MB, from github.com/{manifest['repo']} at commit {manifest['commit'][:10]}")
    print(download.fetch(manifest, entries))
    return 0


def cmd_audit(args) -> int:
    from .data import download
    from .data.financebench import load_human_labels, load_questions
    from .eval.humans import audit

    print(json.dumps(audit(load_questions(), load_human_labels(), download.load_manifest()), indent=1))
    return 0


def cmd_humans(args) -> int:
    from .data.financebench import load_human_labels
    from .eval.humans import paper_table

    print(f"{'model':22s} {'setting':20s} {'correct':>8s} {'wrong':>6s} {'refused':>8s}")
    for (model, mode), counts in sorted(paper_table(load_human_labels()).items()):
        print(f"{model:22s} {mode:20s} {counts['correct']:8d} {counts['incorrect']:6d} {counts['refused']:8d}")
    return 0


def cmd_numeric_check(args) -> int:
    from .data.financebench import load_human_labels, load_questions
    from .eval.humans import numeric_agreement

    questions = {q.id: q for q in load_questions()}
    labels = load_human_labels()
    for split in (None, "dev", "test"):
        r = numeric_agreement(labels, questions, split)
        print(
            f"{split or 'all':4s} n={r['n']:4d} agreement={100 * r['agreement']:.1f}% "
            f"({100 * r['lo']:.1f} to {100 * r['hi']:.1f}) kappa={r['kappa']:.3f} table={r['table']}"
        )
    return 0


def _client(fake: bool, model: str = config.ANSWER_MODEL):
    if fake:
        from .llm import FakeLLM

        def respond(req):
            if req.json_schema is None:
                return "This is a dry run.\nFinal answer: NOT RUN"
            if "enough" in req.json_schema["properties"]:
                return json.dumps({"enough": True, "better_query": ""})
            return json.dumps({"reasoning": "This is a dry run.", "found": False, "final_answer": "", "pages": []})

        return FakeLLM(responder=respond, ledger=None)
    from .llm import client_for

    return client_for(model)


def _ledger(model: str):
    """The ledger a model's spending is counted in: Google Cloud credit for Gemini, the Claude budget for the rest."""
    from .ledger import Ledger, google_ledger

    return google_ledger() if model.startswith("gemini-") else Ledger()


def _cost(system, questions, reps: int) -> tuple[float, float, list[int]]:
    """A rough cost and an upper bound for one run, and the size of each prompt."""
    from .pricing import Usage, cost_usd, worst_case_cost_usd

    requests = [r for q in questions for r in system.plan(q)]
    chars = [r.prompt_chars() for r in requests]
    # Three characters per token and 600 output tokens are rough guesses. The pilot replaces them.
    typical = sum(cost_usd(system.model, Usage(input_tokens=c // 3, output_tokens=600)) for c in chars) * reps
    worst = sum(worst_case_cost_usd(system.model, c, r.max_tokens) for c, r in zip(chars, requests)) * reps
    return typical, worst, chars


def _plan(system, questions, reps: int) -> None:
    typical, worst, chars = _cost(system, questions, reps)
    ledger = _ledger(system.model)
    print(f"Plan: {len(questions)} questions x {reps} runs on {system.model}")
    print(f"  prompt size: median {sorted(chars)[len(chars) // 2]:,d} characters, largest {max(chars):,d}")
    print(f"  rough cost ${typical:.4f}, cannot exceed ${worst:.4f}")
    print(f"  ledger: ${ledger.spent():.4f} spent of ${ledger.cap_usd:.2f}, ${ledger.remaining():.4f} left")


def cmd_run(args) -> int:
    from .eval.runner import answer_run
    from .systems.bounds import SYSTEMS

    questions = _questions(args.split, args.limit)
    if args.system in SYSTEMS:
        system = SYSTEMS[args.system](model=args.model)
    else:
        from .rag.ingest import Index
        from .rag.retrieve import Retriever
        from .rag.systems import AgenticRAG, SimpleRAG

        system = {"rag": SimpleRAG, "agent": AgenticRAG}[args.system](Retriever(Index()), model=args.model)
    _plan(system, questions, args.reps)
    if not (args.confirm or args.fake):
        print("Nothing was sent. Add --confirm to run it, or --fake to test the plumbing for free.")
        return 0
    runs_dir = config.RUNS_DIR / "_dry" if args.fake else config.RUNS_DIR
    run_dir = answer_run(
        system, questions, _client(args.fake, args.model), reps=args.reps, line=args.line, runs_dir=runs_dir, concurrency=args.concurrency
    )
    print(f"Answers are in {run_dir}")
    return 0


def cmd_grade(args) -> int:
    from .data.financebench import load_questions
    from .eval.judge import Judge, build_request, rubric_version
    from .eval.runner import UNGRADED, _read_jsonl, grade_run, grades_path
    from .pricing import Usage, cost_usd, worst_case_cost_usd

    run_dir = Path(args.run_dir)
    if "_dry" in run_dir.resolve().parts:
        print("This is a dry run made with the stand-in model. It is not graded with a paid judge.")
        return 2
    questions = {q.id: q for q in load_questions()}
    done = {(g["case_id"], g["rep"]) for g in _read_jsonl(grades_path(run_dir, f"{rubric_version()}:{args.model}"))}
    todo = [r for r in _read_jsonl(run_dir / "answers.jsonl") if r["status"] not in UNGRADED and (r["case_id"], r["rep"]) not in done]
    if args.rep is not None:
        todo = [r for r in todo if r["rep"] == args.rep]
    chars = []
    for row in todo:
        q = questions[row["case_id"]]
        chars.append(build_request(q.question, q.answer, q.justification, row["answer"], args.model).prompt_chars())
    typical = sum(cost_usd(args.model, Usage(input_tokens=c // 3, output_tokens=200)) for c in chars)
    worst = sum(worst_case_cost_usd(args.model, c, 2048) for c in chars)
    print(f"Plan: grade {len(todo)} answers of {run_dir.name} with {args.model}. Rough cost ${typical:.2f}, cannot exceed ${worst:.2f}.")
    if not args.confirm:
        print("Nothing was sent. Add --confirm to run it.")
        return 0
    path = grade_run(run_dir, questions, Judge(_client(False), model=args.model), concurrency=args.concurrency, only_rep=args.rep)
    print(f"Grades are in {path}")
    return 0


def cmd_judge_compare(args) -> int:
    """How often two judges give the same label to the same answers. No model is called, so this costs nothing."""
    from .eval.judge import rubric_version
    from .eval.report import judge_overlap

    s = judge_overlap([Path(p) for p in args.run_dirs], f"{rubric_version()}:{args.reference}", f"{rubric_version()}:{args.candidate}")
    if not s["n"]:
        print(f"No answer was graded by both {args.reference} and {args.candidate}. Grade the runs with each of them first.")
        return 1
    print(f"{s['n']} answers from {len(s['runs'])} runs were graded by both judges.")
    print(f"agreement={100 * s['agreement']:.1f}% ({100 * s['lo']:.1f} to {100 * s['hi']:.1f}) kappa={s['kappa']:.3f}")
    print(f"correct by {args.reference} {100 * s['correct_by_reference']:.1f}%, by {args.candidate} {100 * s['correct_by_candidate']:.1f}%")
    print(f"rows are {args.reference}, columns are {args.candidate}, in the order {s['labels']}: {s['table']}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=1))
    print(f"{len(s['disagreements'])} disagreements are listed in {out}")
    return 0


def cmd_judge_smoke(args) -> int:
    """Four probes the judge must get right before it grades anything."""
    from .data.financebench import load_questions
    from .eval.judge import Judge

    if not args.confirm:
        print(f"Plan: three short requests to {args.model}, a few cents at most. Nothing was sent. Add --confirm to run it.")
        return 0
    judge = Judge(_client(False), model=args.model)
    q = next(q for q in load_questions() if q.id == "financebench_id_03029")
    probes = [
        ("the reference answer itself", q.answer, "correct"),
        ("an empty answer", "", "incorrect"),
        ("a refusal", "I don't know. I do not have access to that filing.", "refused"),
        ("a confident wrong number", "The FY2018 capital expenditure was $2,345 million.\nFinal answer: $2,345 million", "incorrect"),
    ]
    failed = 0
    for name, candidate, expected in probes:
        verdict = judge.grade(q.question, q.answer, q.justification, candidate, run_id="judge-smoke")
        ok = verdict.label == expected
        failed += not ok
        print(f"{'pass' if ok else 'FAIL'}  {name}: expected {expected}, got {verdict.label}. {verdict.reason}")
    return 1 if failed else 0


def cmd_judge_check(args) -> int:
    from .data.financebench import load_human_labels, load_questions
    from .eval.humans import judge_agreement, sample_labels
    from .eval.judge import Judge, build_request
    from .pricing import Usage, cost_usd, worst_case_cost_usd

    questions = {q.id: q for q in load_questions()}
    split = None if args.split == "all" else args.split
    sample = sample_labels(load_human_labels(), questions, args.n, split)
    chars = []
    for h in sample:
        q = questions[h.question_id]
        chars.append(build_request(q.question, q.answer, q.justification, h.model_answer, args.model).prompt_chars())
    # Three characters per token and 200 output tokens are rough guesses. The first rows replace them.
    typical = sum(cost_usd(args.model, Usage(input_tokens=c // 3, output_tokens=200)) for c in chars)
    worst = sum(worst_case_cost_usd(args.model, c, 2048) for c in chars)
    print(f"Plan: grade {len(sample)} human-labelled answers ({args.split} questions) with {args.model}.")
    print(f"  rough cost ${typical:.2f}, cannot exceed ${worst:.2f}")
    if not args.confirm:
        print("Nothing was sent. Add --confirm to run it.")
        return 0
    judge = Judge(_client(False), model=args.model)
    out_dir = config.RUNS_DIR / f"judge-check-{args.split}-{judge.grader_id.replace(':', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    s = judge_agreement(judge, sample, questions, out_dir)
    print(f"n={s['n']} agreement={100 * s['agreement']:.1f}% ({100 * s['lo']:.1f} to {100 * s['hi']:.1f}) kappa={s['kappa']:.3f}")
    print(f"accuracy by humans {100 * s['accuracy_by_humans']:.1f}%, by judge {100 * s['accuracy_by_judge']:.1f}%, cost ${s['cost_usd']:.4f}")
    print(f"rows are human label, columns are judge label, in the order {s['labels']}: {s['table']}")
    print(f"{len(s['disagreements'])} disagreements are listed in {out_dir / 'summary.json'}")
    return 0


def cmd_report(args) -> int:
    from .eval.report import compare, load_run, render, same_grader, summarize

    runs = [load_run(Path(p), args.grader) for p in args.run_dirs]
    same_grader(runs)
    summaries = [summarize(r, allow_partial=args.allow_partial) for r in runs]
    comparisons = [compare(a, b) for a, b in zip(runs, runs[1:])]
    text = render(summaries, comparisons)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        out.with_suffix(".json").write_text(json.dumps({"runs": summaries, "comparisons": comparisons}, indent=1))
    return 0


def cmd_breakdown(args) -> int:
    """Split a system's score by what the model knew without documents, and by whether the evidence page was read. No model is called."""
    from .eval.report import breakdown, load_run, render_breakdown

    result = breakdown(load_run(Path(args.floor_run), args.grader), load_run(Path(args.system_run), args.grader))
    text = render_breakdown(result)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    return 0


def cmd_stats_check(args) -> int:
    from .eval import simulate

    for n, reps in ((150, 3), (150, 1), (99, 3)):
        print(json.dumps(simulate.check(args.sims, n, reps)))
    return 0


def cmd_ingest(args) -> int:
    from .rag.ingest import ingest

    print(ingest())
    return 0


def cmd_retrieval_check(args) -> int:
    """Score each search method on the evidence pages. No model is called, so this costs nothing."""
    from .eval.metrics import retrieval_scores
    from .rag.ingest import Index
    from .rag.retrieve import MODES, Retriever

    questions = _questions(args.split, args.limit)
    retriever = Retriever(Index())
    from .rag.graph import PAGES

    columns = (f"doc_hit@{PAGES}", "hit@1", "hit@5", f"hit@{PAGES}", f"recall@{PAGES}", "mrr")
    # Each row adds one thing to the row before it: where to search, then which pages to put first.
    settings = [(mode, False, False) for mode in MODES] + [(mode, True, False) for mode in MODES] + [(mode, True, True) for mode in MODES]
    rows = {}
    for mode, routed, pin in settings:
        scores = []
        for q in questions:
            pages = retriever.find(q.question, mode, n_pages=20, routed=routed, pin=pin)
            scores.append(retrieval_scores(q.gold_pages, [(p.doc, p.page) for p in pages], ks=(1, 5, PAGES)))
        name = mode + (", routed" if routed else "") + (", statements first" if pin else "")
        rows[name] = {c: sum(s[c] for s in scores) / len(scores) for c in columns}
    lines = [
        f"Retrieval on {len(questions)} questions ({args.split}), {len(retriever.chunks):,d} chunks in the index. The model is handed {PAGES} pages.",
        "",
        f"| Search | Right filing in top {PAGES} | Evidence page first | In top 5 | In top {PAGES} | Evidence pages found in top {PAGES} | MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{100 * r[c]:.1f}%" if c != "mrr" else f"{r[c]:.3f}" for c in columns) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        out.with_suffix(".json").write_text(json.dumps({"questions": len(questions), "split": args.split, "rows": rows}, indent=1))
    return 0


def cmd_updates_demo(args) -> int:
    from .rag.updates import demo

    print(json.dumps(demo(args.filing), indent=1))
    return 0


def _study(edit: str, model: str, policies: tuple[str, ...] | None):
    """The cases of one test, and one system for each of its settings."""
    from .data.financebench import load_questions

    if edit == "index":  # the test through the real index. It first indexes each filing as first filed and as amended
        from .rag import amended_index
        from .rag.retrieve import Retriever

        index, cases = amended_index.build(load_questions())
        return cases, lambda chosen: amended_index.systems(Retriever(index), chosen, model, policies)
    from .rag.stale import build_cases, systems

    return build_cases(load_questions(), edit), lambda chosen: systems(chosen, model, policies, edit)


def cmd_stale_run(args) -> int:
    """Ask every case of one test under each of its settings. One run per setting."""
    from .eval.runner import answer_run

    cases, build = _study(args.edit, args.model, tuple(args.policy) if args.policy else None)
    if args.limit:
        cases = cases[: args.limit]
    questions = [c.question for c in cases]
    todo = build(cases)
    costs = [_cost(system, questions, args.reps) for system in todo]
    print(f"Plan: {len(cases)} cases ({args.edit}) x {args.reps} runs x {len(todo)} settings on {args.model}")
    for system, (typical, worst, _) in zip(todo, costs):
        print(f"  {system.name:32s} rough cost ${typical:.4f}, cannot exceed ${worst:.4f}")
    print(f"  total rough cost ${sum(c[0] for c in costs):.2f}, cannot exceed ${sum(c[1] for c in costs):.2f}")
    if not (args.confirm or args.fake):
        print("Nothing was sent. Add --confirm to run it, or --fake to test the plumbing for free.")
        return 0
    client = _client(args.fake, args.model)
    runs_dir = config.RUNS_DIR / "_dry" if args.fake else config.RUNS_DIR
    for system in todo:
        answer_run(system, questions, client, reps=args.reps, line=args.line, runs_dir=runs_dir, concurrency=args.concurrency)
    print(f"Answers are in {runs_dir}. `footnote stale-report` builds the table.")
    return 0


def cmd_stale_report(args) -> int:
    """Count which amount came back in each setting of one test. No model is called, so this costs nothing."""
    from .rag.stale import RESULT_NAMES, render, summarize

    cases, _ = _study(args.edit, args.model, None)
    summary = summarize(cases, args.model, args.edit)
    text = render(summary)
    print(text)
    out = Path(args.out or config.RESULTS_DIR / f"{RESULT_NAMES[args.edit]}-{args.model}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    return 0


def cmd_scorecard(args) -> int:
    """One table of every test and every model that has results. No model is called, so this costs nothing."""
    from .eval.charts import MODEL_NAMES
    from .rag.stale import render_scorecard, scorecard

    found, rows = scorecard(config.RESULTS_DIR, tuple(MODEL_NAMES))
    if not found:
        print("No results yet. Run `footnote stale-run` and `footnote stale-report` first.")
        return 1
    text = render_scorecard(found, rows)
    print(text)
    (config.RESULTS_DIR / "scorecard.md").write_text(text)
    return 0


def cmd_charts(args) -> int:
    """Draw the charts from the result files. No model is called, so this costs nothing."""
    from .eval import charts

    made = []
    report = Path(args.report)
    if report.exists():
        data = charts.load(report)
        first = data["runs"][0]
        subtitle = f"{first['n_cases']} FinanceBench questions, {first['reps']} runs each, answered by {first['model']}."
        note = f"Graded by {first['grader'].split('_')[-1]}. Each share is a mean over questions."
        made.append(charts.write(charts.ladder_chart(data, subtitle, note), report.with_suffix(".svg")))
    for path in sorted(config.RESULTS_DIR.glob("retrieval-*.json")):
        made.append(charts.write(charts.retrieval_chart(charts.load(path)), path.with_suffix(".svg")))
    from .rag.stale import RESULT_NAMES

    for path in sorted(p for name in RESULT_NAMES.values() for p in config.RESULTS_DIR.glob(f"{name}-*.json")):
        data = charts.load(path)
        for group, block in data["groups"].items():
            if block["read_correctly_before"] < 20:
                continue  # too few cases for a chart. The table still shows them.
            note = f"{block['read_correctly_before']} statement lines from real filings, each read correctly on its own page first. Three runs per line."
            made.append(charts.write(charts.stale_chart(data, group, note), path.with_name(f"{path.stem}-{group}.svg")))
    scorecard = charts.scorecard_chart(config.RESULTS_DIR)
    if scorecard:
        made.append(charts.write(scorecard, config.RESULTS_DIR / "scorecard.svg"))
    # The same systems on other answering models, when `footnote report --out results/results-<model>.md` was run for them.
    reports = {charts.load(report)["runs"][0]["model"]: charts.load(report)} if report.exists() else {}
    for path in sorted(config.RESULTS_DIR.glob("results-*.json")):
        reports[charts.load(path)["runs"][0]["model"]] = charts.load(path)
    if len(reports) > 1:
        made.append(charts.write(charts.models_chart(reports), config.RESULTS_DIR / "models.svg"))
    for path in made:
        print(f"wrote {path}")
    return 0 if made else 1


def cmd_budget(args) -> int:
    from .ledger import Ledger

    ledger = Ledger()
    print(f"{'line':16s} {'cap':>7s} {'spent':>9s} {'calls':>6s}")
    for row in ledger.summary():
        print(f"{row['line']:16s} {row['cap_usd']:7.2f} {row['spent_usd']:9.4f} {row['calls']:6d}")
    print(f"{'total':16s} {ledger.cap_usd:7.2f} {ledger.spent():9.4f}")
    print(f"reserved by calls in flight: ${ledger.reserved():.4f}. Left: ${ledger.remaining():.4f}.")
    health = ledger.health()
    if health["unconfirmed_calls"]:
        print(f"{health['unconfirmed_calls']} calls failed after they were sent. They are counted at ${health['unconfirmed_usd']:.4f}, their worst case.")
    if health["overrun_calls"]:
        print(f"WARNING: {health['overrun_calls']} calls cost more than was reserved for them, by up to ${health['largest_overrun_usd']:.4f}.")
    if config.GOOGLE_LEDGER_PATH.exists():
        from .ledger import google_ledger

        google = google_ledger()
        print(f"Google Cloud credit, for everything on Gemini: ${google.spent():.4f} spent of the ${google.cap_usd:.2f} cap, kept in its own ledger.")
        for row in google.summary():
            print(f"  {row['line']:14s} {row['cap_usd']:7.2f} {row['spent_usd']:9.4f} {row['calls']:6d}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="footnote", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="fetch or verify the benchmark files")
    p.add_argument("action", choices=["fetch", "verify"])
    p.add_argument("--pdfs", choices=["none", "needed", "all"], default="none", help="also take the filings as PDF")
    p.set_defaults(func=cmd_data)

    sub.add_parser("audit", help="check the question set for duplicates, balance and gaps").set_defaults(func=cmd_audit)
    sub.add_parser("humans", help="recount the benchmark paper's table from its human labels").set_defaults(func=cmd_humans)
    sub.add_parser("numeric-check", help="score the free numeric check against the human labels").set_defaults(func=cmd_numeric_check)

    p = sub.add_parser("run", help="answer the questions with one system")
    p.add_argument("system", choices=["floor", "rag", "agent", "ceiling"])
    p.add_argument("--model", default=config.ANSWER_MODEL, help="the model that answers. A Gemini model is paid from Google Cloud credit")
    p.add_argument("--reps", type=int, default=config.DEFAULT_REPS)
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--line", default="development", choices=sorted(config.BUDGET_LINES))
    p.add_argument("--confirm", action="store_true", help="send the requests. Without it only the plan is printed")
    p.add_argument("--fake", action="store_true", help="use a stand-in model that costs nothing")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("grade", help="grade the answers of a run with the model judge")
    p.add_argument("run_dir")
    p.add_argument("--model", default=config.JUDGE_MODEL)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--rep", type=int, help="grade only this repeat of every question, as a sample. Repeats are numbered from 0")
    p.add_argument("--confirm", action="store_true", help="send the requests. Without it only the plan is printed")
    p.set_defaults(func=cmd_grade)

    p = sub.add_parser("judge-compare", help="compare two judges on the answers both graded, at no cost")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--reference", default=config.JUDGE_MODEL, help="the judge the other one is measured against")
    p.add_argument("--candidate", default=config.ANSWER_MODEL)
    p.add_argument("--out", default=str(config.RESULTS_DIR / "judge-compare.json"))
    p.set_defaults(func=cmd_judge_compare)

    p = sub.add_parser("judge-smoke", help="four probes the judge must pass")
    p.add_argument("--model", default=config.JUDGE_MODEL)
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_judge_smoke)

    p = sub.add_parser("judge-check", help="score the model judge against the human labels")
    p.add_argument("--n", type=int, default=320)
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--model", default=config.JUDGE_MODEL)
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_judge_check)

    p = sub.add_parser("report", help="build the results table from graded runs")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", default=str(config.RESULTS_DIR / "results.md"))
    p.add_argument("--grader", help="the judge model whose grades to use, when a run was graded by more than one judge")
    p.add_argument("--allow-partial", action="store_true", help="print a table for an unfinished run, with its gaps marked")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("breakdown", help="split a system's score by what the model already knew and by what search found, at no cost")
    p.add_argument("floor_run", help="the run without documents")
    p.add_argument("system_run", help="the run of the system to break down")
    p.add_argument("--grader", help="the judge model whose grades to use")
    p.add_argument("--out", default=str(config.RESULTS_DIR / "breakdown.md"))
    p.set_defaults(func=cmd_breakdown)

    p = sub.add_parser("stats-check", help="check the intervals and the paired test on simulated benchmarks")
    p.add_argument("--sims", type=int, default=2000)
    p.set_defaults(func=cmd_stats_check)

    sub.add_parser("ingest", help="index the filings: pages, chunks, vectors").set_defaults(func=cmd_ingest)

    p = sub.add_parser("retrieval-check", help="score each search method on the evidence pages, at no cost")
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", default=str(config.RESULTS_DIR / "retrieval.md"))
    p.set_defaults(func=cmd_retrieval_check)

    p = sub.add_parser("updates-demo", help="amend one filing and show the index follow")
    p.add_argument("--filing", default="3M_2018_10K")
    p.set_defaults(func=cmd_updates_demo)

    p = sub.add_parser("stale-run", help="ask about amended pages under each update policy")
    p.add_argument("--model", default=config.ANSWER_MODEL)
    p.add_argument("--edit", choices=["single", "moved", "wrong", "injected", "index"], default="single", help="the test. single changes one amount, moved shifts an amount between two lines, wrong hands over another company's page, injected adds an instruction to the page, index runs the moved test through the real index")
    p.add_argument("--policy", action="append", help="one setting of the test only. Repeat it for more. All of them when left out")
    p.add_argument("--reps", type=int, default=config.DEFAULT_REPS)
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--line", default="changing_docs", choices=sorted(config.BUDGET_LINES))
    p.add_argument("--confirm", action="store_true", help="send the requests. Without it only the plan is printed")
    p.add_argument("--fake", action="store_true", help="use a stand-in model that costs nothing")
    p.set_defaults(func=cmd_stale_run)

    p = sub.add_parser("stale-report", help="count old and new amounts in the answers about amended pages, at no cost")
    p.add_argument("--model", default=config.ANSWER_MODEL)
    p.add_argument("--edit", choices=["single", "moved", "wrong", "injected", "index"], default="single")
    p.add_argument("--out")
    p.set_defaults(func=cmd_stale_report)

    sub.add_parser("scorecard", help="one table of every test and every model, at no cost").set_defaults(func=cmd_scorecard)

    p = sub.add_parser("charts", help="draw the results as SVG charts, at no cost")
    p.add_argument("--report", default=str(config.RESULTS_DIR / "results.json"), help="the JSON file that `footnote report` wrote")
    p.set_defaults(func=cmd_charts)

    sub.add_parser("budget", help="show what has been spent against each cap").set_defaults(func=cmd_budget)

    args = parser.parse_args(argv)
    from .data.financebench import MissingData
    from .ledger import BudgetExceeded
    from .llm import MissingApiKey, SpendLimitReached

    from .eval.report import IncompleteRun
    from .eval.runner import RunAborted, RunLocked

    try:
        return args.func(args)
    except (MissingData, MissingApiKey, BudgetExceeded, SpendLimitReached, IncompleteRun, RunAborted, RunLocked) as err:
        print(f"Stopped: {err}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
