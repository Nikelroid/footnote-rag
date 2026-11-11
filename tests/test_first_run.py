"""What a new checkout does before the benchmark files are there. Nothing is downloaded in these tests."""

import types

import pytest

from footnote import config
from footnote.cli import main
from footnote.data import download, financebench
from footnote.data.financebench import MissingData, load_human_labels, load_questions
from footnote.rag import ingest


def test_loading_without_the_benchmark_files_says_how_to_get_them(tmp_path):
    with pytest.raises(MissingData, match="footnote data fetch"):
        load_questions(tmp_path)
    with pytest.raises(MissingData, match="footnote data fetch"):
        load_human_labels(tmp_path)
    with pytest.raises(MissingData, match="footnote data fetch"):
        ingest.load_pages("3M_2018_10K", {}, tmp_path / "3M_2018_10K.pdf")


def test_a_command_without_the_benchmark_files_stops_with_one_line(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    assert main(["audit"]) == 2
    out = capsys.readouterr().out
    assert out.startswith("Stopped: ") and "footnote data fetch --pdfs needed" in out and "Traceback" not in out


def test_fetching_the_needed_filings_takes_the_question_file_first(monkeypatch, capsys):
    fetched = []

    def fake_fetch(manifest, entries, **kwargs):
        fetched.append([entry["path"] for entry in entries])
        return {"downloaded": len(entries)}

    def questions_once_fetched(root=None):
        if not fetched:  # on a new checkout the question file only exists after the first fetch
            raise MissingData("the question file is not there yet")
        return [types.SimpleNamespace(doc_name="3M_2018_10K")]

    monkeypatch.setattr(download, "fetch", fake_fetch)
    monkeypatch.setattr(financebench, "load_questions", questions_once_fetched)
    assert main(["data", "fetch", "--pdfs", "needed"]) == 0
    first, second = fetched
    assert "data/financebench_open_source.jsonl" in first and not [path for path in first if path.startswith("pdfs/")]
    assert "pdfs/3M_2018_10K.pdf" in second and len(second) == len(first) + 1
