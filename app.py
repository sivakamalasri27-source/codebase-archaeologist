"""Streamlit Web UI application for Codebase Archaeologist.

Plain English:
Provides a simple web interface to ask natural-language questions about why code
in psf/requests was changed. Uses ChromaDB semantic search and Google Gemini (with
structured fallback) to provide answers with commit citations and relevance stats.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure src/ is on Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qa import CHROMA_DIR, ask_detailed  # noqa: E402


# ---------------------------------------------------------------------------
# Cached Resources & Data
# ---------------------------------------------------------------------------

@st.cache_resource
def load_embedding_model() -> Any:
    """Load and cache the sentence-transformers model once across reruns."""
    from index import _load_model
    return _load_model()


@st.cache_data
def get_commit_lookup() -> dict[str, dict[str, str]]:
    """Build a lookup map of all commit metadata indexed in ChromaDB."""
    from index import _get_collection
    col = _get_collection(CHROMA_DIR)
    res = col.get(where={"type": "commit"}, include=["metadatas"])
    lookup: dict[str, dict[str, str]] = {}
    for meta in res.get("metadatas", []):
        if not meta:
            continue
        full_hash = meta.get("hash", "").lower()
        short_hash = meta.get("short_hash", "").lower() or full_hash[:8]
        entry = {
            "hash": short_hash,
            "full_hash": full_hash,
            "date": meta.get("date", "")[:10],
            "author": meta.get("author", "Unknown"),
            "message": meta.get("message", "").strip(),
        }
        if full_hash:
            lookup[full_hash] = entry
        if short_hash:
            lookup[short_hash] = entry
            lookup[short_hash[:7]] = entry
    return lookup


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

# Warm up cached model
load_embedding_model()

st.title("Codebase Archaeologist")
st.write("Ask why code in psf/requests was changed")

# Example question buttons
st.write("**Example questions:**")
col1, col2, col3 = st.columns(3)


def set_query(q: str) -> None:
    st.session_state["query_input"] = q


with col1:
    st.button(
        "_encode_files change",
        on_click=set_query,
        args=("Why was _encode_files changed?",),
        use_container_width=True,
    )

with col2:
    st.button(
        "Regex header validation",
        on_click=set_query,
        args=("Why did Requests change the regex it uses to check whether a header value is valid?",),
        use_container_width=True,
    )

with col3:
    st.button(
        "Type hints mutability",
        on_click=set_query,
        args=("Why were the type hints for request headers changed to deal with mutability?",),
        use_container_width=True,
    )

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""

question: str = st.text_input(
    "Question:",
    key="query_input",
    placeholder="e.g. Why was _encode_files changed?",
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question or select an example above.")
    else:
        with st.spinner("Searching repository history and synthesizing answer..."):
            result = ask_detailed(question.strip(), verbose=False)

        # 1. Answer section
        st.subheader("Answer")
        if result.get("model_used") == "structured":
            st.info("ℹ️ Gemini LLM was unavailable. Displaying structured fallback built from retrieved repository evidence.")

        st.markdown(result["answer"])

        # 2. Retrieval & Abstain details
        st.subheader("Retrieval Details")
        col_abs, col_dist = st.columns(2)

        with col_abs:
            abstained = result.get("abstained_at_retrieval", False)
            if abstained:
                st.error("Tool abstained: **Yes** (distance exceeded threshold)")
            else:
                st.success("Tool abstained: **No** (sufficient evidence found)")

        with col_dist:
            distances = result.get("top3_distances", [])
            dist_str = ", ".join(f"{d:.4f}" for d in distances) if distances else "None"
            st.write(f"**Top-3 retrieval distances:** `{dist_str}`")

        # 3. Table of cited commits
        st.subheader("Cited Commits")
        cited_hashes: list[str] = list(result.get("cited_hashes", []))

        # If fallback answer was used, parse any cited hashes directly from answer text
        if not cited_hashes:
            extracted = re.findall(r"\[([0-9a-fA-F]{7,40})\]", result["answer"])
            cited_hashes = list(dict.fromkeys(extracted))

        if cited_hashes:
            lookup = get_commit_lookup()
            table_rows: list[str] = []
            for h in cited_hashes:
                key = h.strip().lower()
                info = lookup.get(key)
                if info:
                    c_short = info["hash"]
                    c_full = info["full_hash"] or key
                    c_date = info["date"]
                    c_author = info["author"]
                    c_msg = info["message"].replace("|", "\\|")
                else:
                    c_short = key[:8]
                    c_full = key
                    c_date = "—"
                    c_author = "Unknown"
                    c_msg = "(Not in ChromaDB commit index)"

                commit_url = f"https://github.com/psf/requests/commit/{c_full}"
                link_md = f"[{c_short}]({commit_url})"
                table_rows.append(f"| {link_md} | {c_date} | {c_author} | {c_msg} | [{commit_url}]({commit_url}) |")

            table_content = (
                "| Hash | Date | Author | Message | Link |\n"
                "| :--- | :--- | :--- | :--- | :--- |\n"
                + "\n".join(table_rows)
            )
            st.markdown(table_content)
        else:
            st.write("No commit hashes were cited in this answer.")
