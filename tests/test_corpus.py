import pytest

from conftest import make_question
from footnote.corpus.chunking import fixed_size_chunks, page_chunks
from footnote.corpus.edits import apply_edit, make_edit
from footnote.corpus.registry import Chunk, CorpusRegistry, chunk_hash, source_hash


def filing(n_pages=40, changed_page=None, extra=""):
    pages = []
    for p in range(n_pages):
        paragraphs = [f"Page {p} paragraph {k}. " + " ".join(f"item{p}_{k}_{w}" for w in range(60)) for k in range(4)]
        if p == changed_page:
            paragraphs.insert(1, extra)
        pages.append("\n\n".join(paragraphs))
    return pages


def embed(registry):
    """What the indexer does after every update: give each waiting chunk its vector."""
    registry.mark_embedded([h for h, _ in registry.pending_embeddings()])


def search(registry, term, as_of=None):
    """A stand-in for retrieval: every searchable chunk that contains the term."""
    return [c for c in registry.active(as_of=as_of) if term in c.text]


def test_first_version_embeds_everything_once():
    reg = CorpusRegistry()
    chunks = page_chunks(filing(5))
    change = reg.apply("3M_2018_10K", "v1", chunks, "2019-02-07", source="aaa")
    assert change.previous_version is None and change.embed_ratio == 1.0 and len(change.to_embed) == len(chunks)
    assert [c.text for c in reg.active()] == [c.text for c in chunks]
    assert len(reg.pending_embeddings()) == len(chunks)
    reg.mark_embedded(change.to_embed)
    assert reg.pending_embeddings() == []


def test_unchanged_source_is_skipped_before_any_parsing():
    reg = CorpusRegistry()
    reg.apply("doc", "v1", page_chunks(filing(3)), "2020-01-01", source=source_hash(b"same bytes"))
    assert reg.needs_update("doc", source_hash(b"same bytes")) is False
    assert reg.needs_update("doc", source_hash(b"amended bytes")) is True
    assert reg.needs_update("never seen", "x") is True
    assert reg.apply("doc", "v1", page_chunks(filing(3)), "2020-01-01").noop is True


def test_amendment_embeds_only_what_changed_and_retires_the_old_text():
    reg = CorpusRegistry()
    v1 = [Chunk("Capital expenditure was 1,577 million.", 59), Chunk("Net sales were 32,765 million.", 60), Chunk("Risk factors.", 12)]
    v2 = [Chunk("Capital expenditure was 1,612 million.", 59), Chunk("Net sales were 32,765 million.", 60), Chunk("Risk factors.", 12)]
    reg.apply("3M_2018_10K", "10-K", v1, "2019-02-07")
    embed(reg)
    change = reg.apply("3M_2018_10K", "10-K/A", v2, "2019-06-30")
    assert change.previous_version == "10-K"
    assert change.to_embed == [chunk_hash(v2[0].text)] and change.removed == [chunk_hash(v1[0].text)]
    assert len(change.unchanged) == 2 and change.embed_ratio == pytest.approx(1 / 3)
    # Current search sees the new number and can no longer return the old one.
    assert [c.version for c in search(reg, "1,612")] == ["10-K/A"] and search(reg, "1,577") == []
    # The same question asked as of a date before the amendment still gets the old text.
    assert [c.version for c in search(reg, "1,577", as_of="2019-03-01")] == ["10-K"]
    assert search(reg, "1,612", as_of="2019-03-01") == []
    assert search(reg, "1,577", as_of="2019-06-30") == []  # the amendment counts from its own date
    assert [h["version"] for h in reg.history("3M_2018_10K")] == ["10-K", "10-K/A"]
    assert reg.history("3M_2018_10K")[0]["valid_to"] == "2019-06-30"


def test_text_seen_before_is_never_embedded_again():
    reg = CorpusRegistry()
    boilerplate = Chunk("Forward-looking statements involve risks and uncertainties.", 2)
    reg.apply("A_2020_10K", "v1", [boilerplate, Chunk("A specific.", 3)], "2021-01-01")
    waiting = reg.apply("B_2020_10K", "v1", [boilerplate, Chunk("B specific.", 3)], "2021-01-02")
    assert len(waiting.to_embed) == 2  # the shared paragraph has no vector yet, so it is still on the list
    embed(reg)
    other = reg.apply("C_2020_10K", "v1", [boilerplate, Chunk("C specific.", 3)], "2021-01-03")
    assert other.to_embed == [chunk_hash("C specific.")]  # the shared paragraph is reused across filings
    embed(reg)
    reg.apply("A_2020_10K", "v2", [boilerplate, Chunk("A corrected.", 3)], "2021-03-01")
    embed(reg)
    back = reg.apply("A_2020_10K", "v3", [boilerplate, Chunk("A specific.", 3)], "2021-04-01")
    assert back.to_embed == [] and back.added == [chunk_hash("A specific.")]  # a reverted edit costs nothing
    assert chunk_hash("A  specific.\n") == chunk_hash("A specific.")  # whitespace is not a change


def test_withdrawn_document_leaves_search_and_stays_in_history():
    reg = CorpusRegistry()
    reg.apply("doc", "v1", [Chunk("Guidance for the year is 5%.", 1)], "2022-01-01")
    change = reg.retire("doc", "2022-06-01")
    assert len(change.removed) == 1 and reg.active() == [] and reg.current_version("doc") is None
    assert len(reg.active(as_of="2022-03-01")) == 1
    with pytest.raises(KeyError):
        reg.retire("doc", "2022-07-01")


def test_updates_cannot_go_back_in_time_and_repeated_text_is_counted():
    reg = CorpusRegistry()
    reg.apply("doc", "v1", [Chunk("Same line.", 1), Chunk("Same line.", 2), Chunk("Other.", 3)], "2022-05-01")
    embed(reg)
    with pytest.raises(ValueError):
        reg.apply("doc", "v0", [Chunk("Older.", 1)], "2022-04-01")
    change = reg.apply("doc", "v2", [Chunk("Same line.", 1), Chunk("Other.", 3)], "2022-06-01")
    assert len(change.unchanged) == 2 and change.removed == [chunk_hash("Same line.")] and change.to_embed == []


def test_a_failed_update_changes_nothing():
    reg = CorpusRegistry()
    reg.apply("doc", "v1", [Chunk("Stable text.", 1)], "2022-01-01")
    with pytest.raises(Exception):
        reg.apply("doc", "v2", [Chunk("New text.", 1), Chunk("Broken chunk.", None)], "2022-02-01")
    assert reg.current_version("doc")[0] == "v1" and [c.text for c in reg.active()] == ["Stable text."]
    assert reg.pending_embeddings() == [(chunk_hash("Stable text."), "Stable text.")]


def test_a_version_name_is_never_reused_for_different_text():
    reg = CorpusRegistry()
    reg.apply("doc", "10-K", [Chunk("First text.", 1)], "2022-01-01")
    assert reg.apply("doc", "10-K", [Chunk("First text.", 1)], "2022-01-01").noop is True  # the same update again
    with pytest.raises(ValueError, match="already used"):
        reg.apply("doc", "10-K", [Chunk("Other text.", 1)], "2022-02-01")
    reg.apply("doc", "10-K/A", [Chunk("Amended text.", 1)], "2022-03-01")
    with pytest.raises(ValueError, match="already used"):
        reg.apply("doc", "10-K", [Chunk("First text.", 1)], "2022-04-01")  # going back needs a new name too
    assert [c.text for c in reg.active()] == ["Amended text."]


def test_a_withdrawn_document_cannot_return_before_it_left():
    reg = CorpusRegistry()
    reg.apply("doc", "v1", [Chunk("Old.", 1)], "2022-01-01")
    reg.retire("doc", "2022-06-01")
    with pytest.raises(ValueError, match="earlier than"):
        reg.apply("doc", "v2", [Chunk("New.", 1)], "2022-03-01")
    reg.apply("doc", "v2", [Chunk("New.", 1)], "2022-07-01")
    assert [c.text for c in reg.active(as_of="2022-04-01")] == ["Old."] and reg.active(as_of="2022-06-15") == []
    assert [c.text for c in reg.active()] == ["New."]


def test_a_table_page_is_cut_between_lines_never_inside_a_number():
    cells = [f"Line item {i}\n{i},{100 + i:03d},{200 + i:03d}" for i in range(400)]
    page = "\n".join(cells)  # one cell per line and no blank line, like the benchmark's page text
    chunks = page_chunks([page], max_chars=1500)
    assert len(chunks) > 1 and all(len(c.text) <= 1500 for c in chunks)
    assert "\n".join(c.text for c in chunks).replace("\n\n", "\n") == page  # every cut fell on a line end


def test_page_chunks_respect_pages_and_keep_all_text():
    pages = filing(4)
    chunks = page_chunks(pages, max_chars=900)
    for number, page in enumerate(pages):
        mine = [c.text for c in chunks if c.page == number]
        assert "\n\n".join(mine).split() == page.split() and all(len(t) <= 900 for t in mine)
    assert page_chunks(["x" * 2500], max_chars=1000) == [Chunk("x" * 1000, 0), Chunk("x" * 1000, 0), Chunk("x" * 500, 0)]


def test_fixed_size_chunks_overlap_and_cover_the_text():
    chunks = fixed_size_chunks(["a" * 1500, "b" * 1500], size=1000, overlap=100)
    assert [c.page for c in chunks] == [0, 0, 1, 1] and chunks[0].text[-100:] == chunks[1].text[:100]
    with pytest.raises(ValueError):
        fixed_size_chunks(["x"], size=100, overlap=100)


def test_one_inserted_sentence_stays_local_only_with_page_chunks():
    before = filing(40)
    after = filing(40, changed_page=10, extra="The company restated this figure in an amended filing.")
    ratios = {}
    for name, chunker in (("fixed", fixed_size_chunks), ("page", page_chunks)):
        reg = CorpusRegistry()
        reg.apply("doc", "v1", chunker(before), "2023-01-01")
        embed(reg)
        change = reg.apply("doc", "v2", chunker(after), "2023-02-01")
        ratios[name] = change.embed_ratio
        if name == "page":
            new_pages = {c.page for c in reg.active() if c.chunk_hash in set(change.to_embed)}
            assert new_pages == {10}
    assert ratios["page"] < 0.05  # only the chunks of the page that changed
    assert ratios["fixed"] > 0.70  # every chunk after the insertion point has shifted


def test_counterfactual_edit_has_a_known_answer():
    q = make_question(3)
    edit = make_edit(q)
    page = q.evidence[0].full_page_text
    assert edit.old_text == "1,111" and edit.old_text in page and edit.new_text not in page
    assert edit.new_answer != q.answer and edit.new_answer.startswith("$") and edit.new_answer.endswith(".00")
    changed = apply_edit(page.replace("Net cash used", "Total 21,111 and 1,111.5"), edit)
    assert edit.new_text in changed and "(1,111)" not in changed
    assert "21,111" in changed and "1,111.5" in changed  # longer numbers that contain the digits are left alone
    assert make_edit(make_question(3)) == edit  # the same question always gets the same edit


def test_no_edit_when_the_answer_is_not_printed_on_the_page():
    import dataclasses

    q = make_question(1)
    assert make_edit(dataclasses.replace(q, answer="0.66")) is None  # a computed ratio
    assert make_edit(dataclasses.replace(q, answer="Yes, margins improved.")) is None
