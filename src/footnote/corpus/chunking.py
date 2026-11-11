"""Two ways to cut a filing into chunks, and why the difference matters when filings change.

Fixed-size chunking cuts one long string every N characters. Insert a sentence
on page 10 and every later boundary moves, so every later chunk is new text and
has to be embedded again. Page-anchored chunking never lets a chunk cross a
page. The same insertion then changes the chunks of one page only. Pages are
also the unit a citation points to, so nothing is lost by respecting them.
"""

from __future__ import annotations

from .registry import Chunk


def fixed_size_chunks(pages: list[str], size: int = 1000, overlap: int = 100) -> list[Chunk]:
    """The usual baseline. Pages are joined into one string and cut at fixed offsets.

    A chunk can run over a page break. It then carries the page it starts on, so it
    is used here to measure update cost and not to score retrieval by page.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be smaller than the chunk size")
    starts, text = [], ""
    for page in pages:
        starts.append(len(text))
        text += page + "\n\n"
    chunks = []
    page = 0
    for offset in range(0, max(len(text) - overlap, 1), size - overlap):
        piece = text[offset : offset + size]
        if not piece.strip():
            continue
        while page + 1 < len(starts) and starts[page + 1] <= offset:
            page += 1
        chunks.append(Chunk(text=piece, page=page))
    return chunks


def _pieces(block: str, max_chars: int) -> list[str]:
    """Cut a block that is too long at line ends. A line is cut inside only when the line itself is too long."""
    if len(block) <= max_chars:
        return [block]
    out, current = [], ""
    for line in block.split("\n"):
        while len(line) > max_chars:
            if current:
                out.append(current)
                current = ""
            out.append(line[:max_chars])
            line = line[max_chars:]
        if current and len(current) + 1 + len(line) > max_chars:
            out.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        out.append(current)
    return out


def page_chunks(pages: list[str], max_chars: int = 1500) -> list[Chunk]:
    """Chunks that never cross a page. Inside a page, paragraphs are packed up to `max_chars`.

    A table page is often one block with a cell on every line. It is cut between
    lines, so a number is never split in two.
    """
    chunks = []
    for number, page in enumerate(pages):
        current = ""
        for block in (b.strip() for b in page.split("\n\n")):
            for piece in _pieces(block, max_chars) if block else []:
                if current and len(current) + 2 + len(piece) > max_chars:
                    chunks.append(Chunk(text=current, page=number))
                    current = piece
                else:
                    current = f"{current}\n\n{piece}" if current else piece
        if current:
            chunks.append(Chunk(text=current, page=number))
    return chunks
