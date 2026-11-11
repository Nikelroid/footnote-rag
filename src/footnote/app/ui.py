"""A two-tab demo.

    streamlit run src/footnote/app/ui.py

The first tab answers a question. It shows the answer, the pages it cites, and
every page the search handed to the model, so a wrong answer can be traced to
its cause.

The second tab shows what happens when a filing changes. Pick a line that was
amended, pick what the update left in the index, and see which amount comes
back. It is the measurement in `rag/stale.py`, one case at a time.
"""

from __future__ import annotations

import streamlit as st

from footnote import config
from footnote.ledger import BudgetExceeded
from footnote.llm import MissingApiKey

POLICIES = {
    "synced": "The old page was replaced. Only the amended page is in the index.",
    "appended": "The amended page was added and the old one kept. Nothing tells them apart.",
    "dated": "Both pages are kept, and each shows the date it was filed.",
    "dated+rule": "Both pages with their dates, and the instructions say the later filing wins.",
}


@st.cache_resource(show_spinner="Opening the index")
def assistant():
    from footnote.rag.assistant import Assistant

    return Assistant()


@st.cache_resource(show_spinner="Reading the amended pages")
def amended_cases():
    from footnote.data.financebench import default_root, load_questions
    from footnote.rag.stale import build_cases

    if not (default_root() / "data" / "financebench_open_source.jsonl").exists():
        return []
    return build_cases(load_questions(), "moved")


st.set_page_config(page_title="Footnote", layout="wide")
st.title("Footnote")
st.caption("Questions about SEC filings, answered with the page the answer comes from.")
ask_tab, change_tab = st.tabs(["Ask a question", "When a filing changes"])

with ask_tab:
    question = st.text_input("Question", placeholder="What is the FY2018 capital expenditure amount (in USD millions) for 3M?")
    left, right = st.columns(2)
    use_agent = left.toggle("Search again when the first search is weak", value=True)
    search_only = right.toggle("Search only, no model call", value=False)

    if st.button("Ask", type="primary") and question.strip():
        if search_only:
            pages = assistant().search(question)
            st.info("Search only. These are the pages the model would be given.")
        else:
            try:
                with st.spinner("Searching and answering"):
                    result = assistant().ask(question, agent=use_agent)
            except (MissingApiKey, BudgetExceeded) as err:
                st.error(str(err))
                st.stop()
            pages = result["pages"]
            if result["found"]:
                st.success(result["final_answer"])
            else:
                st.warning("Not found in the filings that were searched.")
            st.write(result["answer"])
            cited = ", ".join(f"{c['doc']} page {c['page'] + 1}" for c in result["citations"]) or "none"
            st.caption(f"Cited: {cited}. Searches: {result['searches']}. Model calls: {result['model_calls']}. Cost: ${result['cost_usd']:.5f}. Time: {result['seconds']} s.")
        st.subheader("Pages handed to the model")
        for page in pages:
            with st.expander(f"{page['doc']}, page {page['page'] + 1} (score {page['score']})"):
                st.text(page["text"])

with change_tab:
    cases = amended_cases()
    if not cases:
        st.info("The benchmark files are not downloaded. Run `footnote data fetch` first.")
    else:
        from footnote.rag.stale import ask_once, changes

        st.write(
            "Each case is a real statement page on which an amount was moved from one line to another, the way a "
            "reclassification does in an amended filing. Every total on the page still adds up. The amended amounts are made up."
        )
        case = st.selectbox(
            "Amended line",
            cases,
            format_func=lambda c: f"{c.doc_name}, page {c.page + 1}: {c.question.question.split(chr(39))[1]} ({c.old} became {c.new})",
        )
        policy = st.radio("What the update left in the index", list(POLICIES), format_func=POLICIES.get)
        old_first = st.radio("Which version the search returns first", ["The old page", "The amended page"], horizontal=True) == "The old page"
        st.caption(f"Question: {case.question.question}")
        if st.button("Ask about this line", type="primary"):
            from footnote.llm import client_for

            try:
                with st.spinner("Asking"):
                    result = ask_once(case, client_for(config.ANSWER_MODEL), policy, old_first)
            except (MissingApiKey, BudgetExceeded) as err:
                st.error(str(err))
                st.stop()
            if result["outcome"] == "new":
                st.success(f"The answer gives the new amount, {case.new}.")
            elif result["outcome"] == "old":
                st.error(f"The answer gives the old amount, {case.old}. This is a stale answer.")
            elif result["outcome"] == "both":
                st.warning(f"The answer gives both amounts, the {result['leads']} one first.")
            else:
                st.warning("The answer does not commit to either amount.")
            st.write(result["answer"])
            st.caption(f"Cost: ${result['cost_usd']:.5f}. Time: {result['seconds']} s.")
        with st.expander("What the amendment changed on the page"):
            st.table([{"Line": label, "As first filed": before, "As amended": after} for label, before, after in changes(case)])
