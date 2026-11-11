"""Fetch benchmark files pinned to one commit and verify every byte.

The manifest lists each file with its size and its git blob hash at that commit.
A downloaded file is accepted only when both match, so the data cannot change
under the project without the check failing. Downloaded files are data. They are
parsed as JSON or PDF and nothing in them is ever executed.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

from .. import config

RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
CORE_PREFIXES = ("data/", "results/")
CORE_FILES = ("README.md", "evaluation_playground.ipynb")


def load_manifest(name: str = "financebench") -> dict:
    return json.loads((config.MANIFESTS_DIR / f"{name}.json").read_text())


def dest_root(manifest: dict) -> Path:
    return config.RAW_DIR / manifest["name"].lower()


def core_entries(manifest: dict) -> list[dict]:
    """The questions, the document table and the human-labelled model answers. About 4 MB."""
    return [f for f in manifest["files"] if f["path"].startswith(CORE_PREFIXES) or f["path"] in CORE_FILES]


def pdf_entries(manifest: dict, doc_names: Iterable[str] | None = None) -> list[dict]:
    pdfs = [f for f in manifest["files"] if f["path"].startswith("pdfs/") and f["path"].lower().endswith(".pdf")]
    if doc_names is None:
        return pdfs
    wanted = {f"pdfs/{name}.pdf" for name in doc_names}
    return [f for f in pdfs if f["path"] in wanted]


def _blob_hasher(size: int) -> "hashlib._Hash":
    h = hashlib.sha1()
    h.update(b"blob %d\0" % size)
    return h


def is_verified(entry: dict, root: Path) -> bool:
    path = root / entry["path"]
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    h = _blob_hasher(entry["size"])
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest() == entry["git_blob_sha1"]


def _download(url: str, target: Path, entry: dict) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    h = _blob_hasher(entry["size"])
    received = 0
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        for block in iter(lambda: resp.read(1 << 20), b""):
            received += len(block)
            if received > entry["size"]:
                break
            h.update(block)
            out.write(block)
    if received != entry["size"] or h.hexdigest() != entry["git_blob_sha1"]:
        tmp.unlink(missing_ok=True)
        raise IOError(f"{entry['path']}: the downloaded bytes do not match the manifest")
    tmp.replace(target)


def fetch(manifest: dict, entries: list[dict], log: Callable[[str], None] = print, attempts: int = 3) -> dict:
    root = dest_root(manifest)
    done = skipped = 0
    for entry in entries:
        target = root / entry["path"]
        if is_verified(entry, root):
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = RAW_URL.format(repo=manifest["repo"], commit=manifest["commit"], path=urllib.parse.quote(entry["path"]))
        for attempt in range(1, attempts + 1):
            try:
                _download(url, target, entry)
                break
            except (IOError, OSError) as err:
                if attempt == attempts:
                    raise
                log(f"  retrying {entry['path']} after: {err}")
                time.sleep(2 * attempt)
        done += 1
        log(f"  {entry['size']:>12,d}  {entry['path']}")
    return {"downloaded": done, "already_present": skipped, "bytes": sum(e["size"] for e in entries), "root": str(root)}


def verify(manifest: dict, entries: list[dict]) -> list[str]:
    """Return the paths that are missing or do not match the manifest."""
    root = dest_root(manifest)
    return [e["path"] for e in entries if not is_verified(e, root)]
