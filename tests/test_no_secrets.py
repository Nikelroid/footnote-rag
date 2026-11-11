"""Nothing that git would publish may hold a key, and no source file may be left out. Run before every push."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

# The shapes of real keys. The stand-in keys in the tests do not match any of them.
KEY_SHAPES = {
    "a Claude API key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "a Google API key": re.compile(r"(AIza[0-9A-Za-z_\-]{30,}|AQ\.[0-9A-Za-z_\-]{30,})"),
    "a private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "a bearer token": re.compile(r"Bearer\s+[A-Za-z0-9_\-.]{30,}"),
}


def git(*args: str) -> list[str]:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        pytest.skip("not a git checkout")
    return [line for line in done.stdout.split("\n") if line]


def test_no_file_that_git_would_publish_holds_a_key():
    found = []
    for name in git("ls-files", "--cached", "--others", "--exclude-standard"):
        path = ROOT / name
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        found += [f"{name} looks like it holds {what}" for what, shape in KEY_SHAPES.items() if shape.search(text)]
    assert not found, found


def test_the_key_file_the_ledgers_and_the_raw_runs_are_never_published():
    # The trailing slash names a folder, so the rule is also checked in a fresh clone where the folder does not exist yet.
    ignored = git("check-ignore", ".env", "ledger.sqlite", "ledger-google.sqlite", "runs/", "data/")
    assert set(ignored) == {".env", "ledger.sqlite", "ledger-google.sqlite", "runs/", "data/"}
    assert not [name for name in git("ls-files", "--cached") if Path(name).name == ".env"]


def test_no_source_file_is_hidden_by_an_ignore_rule():
    # A rule such as `data/` also matches src/footnote/data, and the published code then cannot be imported.
    hidden = [name for name in git("ls-files", "--others", "--ignored", "--exclude-standard", "--", "src", "tests") if name.endswith(".py")]
    assert not hidden, hidden
