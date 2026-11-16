"""Question-answering module for Codebase Archaeologist.

Plain English:
Given a natural-language question like "Why was _encode_files changed?",
this module:
1. Searches the ChromaDB index for the most relevant commits and functions.
2. Prints top-3 retrieval distances and checks against ABSTAIN_DISTANCE (0.70).
   If the best hit is farther than that, it returns "Not enough evidence in the repo history"
   without calling the LLM.
3. Builds an evidence context string from commit messages, PR/issue text, and function history.
4. Uses Google Gemini (via official google-genai SDK) to synthesize an answer citing
   commit hashes like [6f205ff4] and prioritizing PR/issue explanations.
5. If Gemini fails (missing key, quota exceeded, API error), it falls back to structured
   synthesis and prints clearly: "LLM not used: <reason>".
   If Gemini returns a model-not-found error, prints the exact error and queries the SDK for
   available models.
6. Validates deduplicated citations into three classification levels:
   a) "in evidence": appears in the retrieved hits (commit hash or in function history text).
   b) "exists in repo": not in retrieved evidence, but is an indexed commit in ChromaDB.
   c) "not found": not in repo index at all (hallucination).
   For structured fallback, prints: "Citation check: n/a (fallback builds citations from retrieved hits)".
7. Tracks execution time and prints a step-by-step breakdown (model load, retrieval, Gemini call)
   if total time exceeds 20 seconds.
"""

from __future__ import annotations

import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Allow importing sibling modules when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from index import search


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

# Model name constant as required
GEMINI_MODEL: str = "gemini-3.5-flash"

# Distance threshold: if the top hit is farther than this, abstain from answering
ABSTAIN_DISTANCE: float = 0.70

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CACHE_PATH = PROJECT_ROOT / "cache_github_requests.json"

TOP_K_COMBINED: int = 10
MAX_CONTEXT_CHARS: int = 6000

# Regex matching commit hash citations: e.g. [6f205ff4] or [6f66281a]
_CITATION_RE = re.compile(r"\[([0-9a-fA-F]{7,40})\]")

# Cache for all repo commit hashes indexed in ChromaDB
_cached_repo_commit_hashes: set[str] | None = None


# ---------------------------------------------------------------------------
# Retrieval & Repo Index Helpers
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    chroma_dir: str | Path = CHROMA_DIR,
    top_k: int = TOP_K_COMBINED,
) -> list[dict[str, Any]]:
    """Search ChromaDB for commits and functions relevant to a question.

    Plain English:
    Runs a combined vector similarity search across commit documents and function
    history documents stored in ChromaDB. The cosine distance measures semantic
    similarity (closer to 0 is more relevant).
    """
    hits = search(question, n=top_k, chroma_dir=chroma_dir)
    return hits


def _get_repo_commit_hashes(chroma_dir: str | Path = CHROMA_DIR) -> set[str]:
    """Return all commit hashes indexed in ChromaDB (cached across calls).

    Plain English:
    Queries ChromaDB for all documents of type 'commit' and collects their
    full and short hashes. Used to check if an LLM citation exists in the repository
    even if it was not in the top-k retrieved search results.
    """
    global _cached_repo_commit_hashes
    if _cached_repo_commit_hashes is not None:
        return _cached_repo_commit_hashes

    try:
        from index import _get_collection
        col = _get_collection(chroma_dir)
        res = col.get(where={"type": "commit"}, include=[])
        hashes: set[str] = set()
        for doc_id in res.get("ids", []):
            if doc_id.startswith("commit:"):
                full_hash = doc_id.split("commit:", 1)[1].strip().lower()
                hashes.add(full_hash)
                hashes.add(full_hash[:8])
                hashes.add(full_hash[:7])
        _cached_repo_commit_hashes = hashes
    except Exception as exc:
        print(f"[qa] Could not load repo commit hashes from ChromaDB: {exc}")
        _cached_repo_commit_hashes = set()

    return _cached_repo_commit_hashes


def _build_context(hits: list[dict[str, Any]]) -> str:
    """Format retrieved hits into numbered evidence blocks for the prompt.

    Plain English:
    Formats each search result as a clearly labeled evidence block containing
    commit hash, author, date, message, modified files, and any attached PR/issue text.
    """
    blocks: list[str] = []
    total_chars = 0

    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        doc_type = meta.get("type", "unknown")

        if doc_type == "commit":
            block = (
                f"[Evidence {i} - Commit]\n"
                f"  Hash:     {meta.get('short_hash', '?')}\n"
                f"  Author:   {meta.get('author', '?')}\n"
                f"  Date:     {meta.get('date', '?')[:10]}\n"
                f"  Message:  {meta.get('message', '')}\n"
                f"  Files:    {meta.get('changed_files', '')[:120]}\n"
                f"  Distance: {hit['distance']:.4f}\n"
                f"  Full text:\n{textwrap.indent(hit['text'][:800], '    ')}"
            )
        elif doc_type == "function":
            block = (
                f"[Evidence {i} - Function]\n"
                f"  Identity: {meta.get('identity', '?')}\n"
                f"  Changes:  {meta.get('num_commits', 0)} time(s)\n"
                f"  Last by:  {meta.get('last_author', '?')} on {meta.get('last_date', '?')}\n"
                f"  Message:  {meta.get('last_message', '')}\n"
                f"  Distance: {hit['distance']:.4f}\n"
                f"  Full text:\n{textwrap.indent(hit['text'][:600], '    ')}"
            )
        else:
            block = f"[Evidence {i} - {doc_type}]\n  {hit['text'][:400]}"

        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        total_chars += len(block)

    return "\n\n".join(blocks)


def _build_llm_prompt(question: str, context: str) -> str:
    """Build the prompt instructing Gemini to act as a codebase archaeologist.

    Plain English:
    Instructs the LLM to strictly base its answer on the provided evidence,
    cite the first 8 characters of commit hashes like [6f205ff4] after every claim,
    prioritize PR or issue text as the reason for changes, and explicitly state
    when the evidence does not explain why without guessing.
    """
    return textwrap.dedent(f"""\
        You are a code historian answering questions about why code was changed
        in the psf/requests library.

        RULES:
        1. Answer ONLY from the evidence blocks provided below. Do not use outside knowledge.
        2. Put the first 8 characters of the commit hash in brackets after EVERY claim, like [6f205ff4].
        3. Prefer PR or issue text as the reason for a change.
        4. Answer the WHY. If the evidence only shows WHAT changed and not the reason, say so in one sentence. Do not list unrelated changes.
        5. If the evidence does not explain WHY, say exactly that: "The evidence does not explain why this change was made." Never guess.
        6. Keep your answer clear, factual, and concise (2-4 paragraphs).

        QUESTION:
        {question}

        EVIDENCE BLOCKS:
        {context}

        ANSWER:
    """).strip()


# ---------------------------------------------------------------------------
# Gemini Integration (Official google-genai SDK)
# ---------------------------------------------------------------------------

def _call_gemini(question: str, context: str) -> tuple[str | None, str | None]:
    """Call Google Gemini using the official google-genai Python SDK.

    Plain English:
    Reads GEMINI_API_KEY from .env and passes the prompt to Gemini using the
    current official Google GenAI SDK (`from google import genai`).
    Returns (answer_text, None) on success, or (None, failure_reason) on failure.
    If the model is not found, queries and displays available models via SDK.
    """
    raw_key = os.getenv("GEMINI_API_KEY", "")
    api_key = raw_key.strip("<> '\"") if raw_key else ""

    if not api_key:
        return None, "GEMINI_API_KEY not found or empty in .env"

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None, "google-genai SDK not installed (run 'pip install google-genai')"

    prompt = _build_llm_prompt(question, context)

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=25000),
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        answer = response.text if response and response.text else ""
        if not answer.strip():
            return None, "Gemini returned an empty response"
        return answer.strip(), None
    except Exception as exc:
        err_msg = str(exc)

        # Requirement 4: If Gemini returns model-not-found error, print exact error
        # and query the list of available models using the SDK.
        if "404" in err_msg or "NOT_FOUND" in err_msg or "not found" in err_msg.lower():
            print(f"\n[qa] [MODEL ERROR] Gemini model '{GEMINI_MODEL}' was not found.")
            print(f"[qa] Exact error: {exc}")
            try:
                print("[qa] Querying available models via SDK (client.models.list())...")
                models = list(client.models.list())
                available_models = [m.name for m in models if getattr(m, "name", None)]
                print(f"[qa] Available models ({len(available_models)}):")
                for m_name in available_models[:15]:
                    print(f"     - {m_name}")
                if len(available_models) > 15:
                    print(f"     ... and {len(available_models) - 15} more")
            except Exception as list_err:
                print(f"[qa] Could not list models via SDK: {list_err}")

        if "API key not valid" in err_msg or "UNAUTHENTICATED" in err_msg:
            err_reason = "API key not valid (401 UNAUTHENTICATED)"
        elif "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
            err_reason = "Quota exceeded (429 RESOURCE_EXHAUSTED)"
        elif "404" in err_msg or "NOT_FOUND" in err_msg:
            err_reason = f"Model '{GEMINI_MODEL}' not found (404 NOT_FOUND)"
        else:
            err_reason = f"{type(exc).__name__}: {err_msg[:120]}"

        return None, err_reason


# ---------------------------------------------------------------------------
# Citation Validation (Three-Level Classification)
# ---------------------------------------------------------------------------

def validate_citations(
    answer: str,
    hits: list[dict[str, Any]],
    chroma_dir: str | Path = CHROMA_DIR,
) -> tuple[list[str], list[str], list[str]]:
    """Classify deduplicated commit hash citations into three levels.

    Plain English:
    Finds every commit hash cited like [6f205ff4] in the answer text, deduplicates them,
    and classifies each into one of three levels:
      a) "in evidence": appears in the retrieved hits (as commit metadata or within
         the text of a commit/function history block).
      b) "exists in repo": not in the retrieved evidence, but is a valid commit hash
         indexed in ChromaDB.
      c) "not found": not found in the index at all (hallucination).

    Returns:
        tuple of (in_evidence_list, exists_in_repo_list, not_found_list)
    """
    cited_raw = _CITATION_RE.findall(answer)
    if not cited_raw:
        return [], [], []

    # Deduplicate citations while preserving order
    unique_cited = list(dict.fromkeys(c.strip().lower() for c in cited_raw))

    # 1. Collect evidence hashes from retrieved hits (metadata + text)
    evidence_hashes: set[str] = set()
    for h in hits:
        meta = h.get("metadata", {})
        if meta.get("short_hash"):
            evidence_hashes.add(meta["short_hash"].lower())
        if meta.get("hash"):
            full = meta["hash"].lower()
            evidence_hashes.add(full)
            evidence_hashes.add(full[:8])
            evidence_hashes.add(full[:7])

        # Scan hit text for any mentioned commit hashes (e.g. in function history)
        hit_text = h.get("text", "")
        for m in _CITATION_RE.findall(hit_text):
            evidence_hashes.add(m.strip().lower())
        for hex_match in re.findall(r"\b[0-9a-fA-F]{7,40}\b", hit_text):
            evidence_hashes.add(hex_match.lower())

    # 2. Collect all commit hashes from repo index in ChromaDB
    repo_hashes = _get_repo_commit_hashes(chroma_dir)

    in_evidence: list[str] = []
    exists_in_repo: list[str] = []
    not_found: list[str] = []

    for c in unique_cited:
        # Check level a: in evidence
        if (
            c in evidence_hashes
            or any(eh.startswith(c) for eh in evidence_hashes)
            or any(c.startswith(eh) for eh in evidence_hashes)
        ):
            in_evidence.append(c)
        # Check level b: exists in repo index
        elif (
            c in repo_hashes
            or any(rh.startswith(c) for rh in repo_hashes)
            or any(c.startswith(rh) for rh in repo_hashes)
        ):
            exists_in_repo.append(c)
        # Level c: not found in the index at all
        else:
            not_found.append(c)

    return in_evidence, exists_in_repo, not_found


# ---------------------------------------------------------------------------
# Structured Synthesis (Always works without an LLM)
# ---------------------------------------------------------------------------

def _synthesise_structured(
    question: str,
    hits: list[dict[str, Any]],
) -> str:
    """Build a well-organized answer directly from search hits without any LLM.

    Plain English:
    Groups search results into commits and functions, extracts commit hashes,
    authors, dates, and PR descriptions, and outputs a formatted markdown summary.
    Guarantees summary and commit content are non-empty.
    """
    if not hits:
        return f"No relevant evidence found for: \"{question}\""

    lines: list[str] = []
    lines.append(f"## {question}\n")

    commit_hits = [h for h in hits if h["metadata"].get("type") == "commit"]
    func_hits   = [h for h in hits if h["metadata"].get("type") == "function"]

    # --- Commit evidence ---
    if commit_hits:
        lines.append("### Relevant Commits\n")
        for rank, hit in enumerate(commit_hits, 1):
            m = hit["metadata"]
            short_hash = m.get("short_hash", "?")
            author = m.get("author", "Unknown author")
            date = m.get("date", "")[:10]
            message = m.get("message", "").strip() or "(no commit message recorded)"
            files = m.get("changed_files", "")
            dist = hit["distance"]

            lines.append(f"**{rank}. [{short_hash}]** ({date}, {author}) - relevance {1 - dist:.0%}")
            lines.append(f"   {message}")

            pr_context = _extract_pr_context(hit["text"])
            if pr_context:
                lines.append(f"   **Context:** {pr_context}")

            if files:
                lines.append(f"   Files: `{files[:100]}`")
            lines.append("")

    # --- Function evidence ---
    if func_hits:
        lines.append("### Related Functions\n")
        for rank, hit in enumerate(func_hits, 1):
            m = hit["metadata"]
            identity = m.get("identity", "?")
            num_commits = m.get("num_commits", 0)
            last_author = m.get("last_author", "?")
            last_date = m.get("last_date", "?")
            last_msg = m.get("last_message", "")
            dist = hit["distance"]

            lines.append(f"**{rank}. `{identity}`** - changed {num_commits}x, relevance {1 - dist:.0%}")
            lines.append(f"   Last modified by {last_author} on {last_date}: {last_msg}")
            lines.append("")

    # --- Summary ---
    # Ensure summary is never empty or a single word
    lines.append("### Summary\n")
    if commit_hits:
        top = commit_hits[0]["metadata"]
        short_h = top.get("short_hash", "?")
        author_h = top.get("author", "Unknown author")
        date_h = top.get("date", "")[:10]
        msg_h = top.get("message", "").strip() or "(no commit message recorded)"
        lines.append(
            f"The most relevant change is commit **[{short_h}]** "
            f"by {author_h} on {date_h}: *\"{msg_h}\"*."
        )
        pr_ctx = _extract_pr_context(commit_hits[0]["text"])
        if pr_ctx:
            lines.append(f"\n{pr_ctx}")
    elif func_hits:
        top_func = func_hits[0]["metadata"]
        lines.append(
            f"No direct commit matches found. The most relevant function is "
            f"`{top_func.get('identity', '?')}`, modified {top_func.get('num_commits', 0)} "
            f"time(s), last by {top_func.get('last_author', '?')} on {top_func.get('last_date', '?')}."
        )
    else:
        print("[qa] [ERROR] Summary is empty: no commit or function hits available.")
        lines.append("[ERROR: Summary could not be constructed due to missing hits.]")

    return "\n".join(lines)


def _extract_pr_context(doc_text: str) -> str:
    """Extract PR or Issue descriptions from a document's full text.

    Plain English:
    Uses regex to extract attached 'PR #NNN' or 'Issue #NNN' snippets from
    the indexed document text.
    """
    pr_pattern = re.compile(
        r"((?:PR|Issue|Linked (?:PR|Issue)) #\d+:.*?)(?=\n(?:PR|Issue|Linked|Commit)|$)",
        re.DOTALL,
    )
    matches = pr_pattern.findall(doc_text)
    if not matches:
        return ""

    snippets: list[str] = []
    for match in matches[:3]:
        clean = match.strip().replace("\n", " ")
        if len(clean) > 300:
            clean = clean[:297] + "..."
        snippets.append(clean)

    return " | ".join(snippets)


# ---------------------------------------------------------------------------
# ask_detailed: Structured dict result (ask() wraps this)
# ---------------------------------------------------------------------------

def ask_detailed(
    question: str,
    chroma_dir: str | Path = CHROMA_DIR,
    top_k: int = TOP_K_COMBINED,
    force_structured: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Answer a question and return a structured dict with all metadata.

    Plain English:
    Does exactly what ask() does internally, but instead of printing everything
    and returning only the answer string, it returns a dict with every piece
    of data the benchmark needs: answer text, model used, citation counts,
    top-3 retrieval distances, abstain flag, and elapsed time.

    Keys returned:
        answer              - full answer string
        model_used          - "gemini", "structured", or "abstained"
        cited_hashes        - deduplicated hashes cited in the answer (LLM only)
        in_evidence         - count of hashes in the retrieved hits
        exists_in_repo      - count of hashes in ChromaDB but not in hits
        not_found           - count of hashes not in repo at all (hallucinations)
        top3_distances      - list of up to 3 retrieval distances
        abstained_at_retrieval - True if answer was skipped due to distance
        seconds             - total elapsed time
    """
    t_start = time.time()
    t_model_load = 0.0
    t_retrieval = 0.0
    t_gemini = 0.0

    if verbose:
        print(f"\n[qa] Question: \"{question}\"")

    # Step 0: Ensure embedding model is loaded
    from index import _model, _load_model
    if _model is None:
        t_ml_start = time.time()
        _load_model()
        t_model_load = time.time() - t_ml_start

    # Step 1: Retrieve hits
    t_ret_start = time.time()
    hits = retrieve(question, chroma_dir=chroma_dir, top_k=top_k)
    t_retrieval = time.time() - t_ret_start

    top3_distances = [round(h["distance"], 4) for h in hits[:3]]

    if verbose:
        print(f"[qa] Top-3 distances: {top3_distances}")

    # Empty hits → abstain
    if not hits:
        elapsed = time.time() - t_start
        return {
            "answer": "Not enough evidence in the repo history",
            "model_used": "abstained",
            "cited_hashes": [],
            "in_evidence": 0,
            "exists_in_repo": 0,
            "not_found": 0,
            "top3_distances": [],
            "abstained_at_retrieval": True,
            "seconds": round(elapsed, 2),
        }

    # Step 2: Distance-based abstain check
    best_dist = hits[0]["distance"]
    if best_dist > ABSTAIN_DISTANCE:
        if verbose:
            print(f"[qa] Best distance {best_dist:.4f} > ABSTAIN_DISTANCE ({ABSTAIN_DISTANCE:.2f}). Abstaining.")
        elapsed = time.time() - t_start
        return {
            "answer": "Not enough evidence in the repo history",
            "model_used": "abstained",
            "cited_hashes": [],
            "in_evidence": 0,
            "exists_in_repo": 0,
            "not_found": 0,
            "top3_distances": top3_distances,
            "abstained_at_retrieval": True,
            "seconds": round(elapsed, 2),
        }

    # Step 3: Build context and attempt Gemini
    context = _build_context(hits)
    answer: str | None = None
    backend_used = "structured"

    if not force_structured:
        t_gem_start = time.time()
        answer, failure_reason = _call_gemini(question, context)
        t_gemini = time.time() - t_gem_start
        if answer is not None:
            backend_used = "gemini"
        else:
            if verbose:
                print(f"[qa] LLM not used: {failure_reason}")

    # Step 4: Fallback to structured synthesis
    if answer is None:
        answer = _synthesise_structured(question, hits)
        backend_used = "structured"

    # Step 5: Ensure answer is non-empty
    cleaned = answer.strip() if answer else ""
    if not cleaned or len(cleaned.split()) <= 1:
        answer = f"[ERROR: Answer generation produced an empty or cut-off response for query: {question}]"
    else:
        answer = cleaned

    # Step 6: Citation classification (LLM only)
    cited_hashes: list[str] = []
    in_ev: list[str] = []
    in_repo: list[str] = []
    not_fnd: list[str] = []

    if backend_used == "gemini":
        in_ev, in_repo, not_fnd = validate_citations(answer, hits, chroma_dir=chroma_dir)
        cited_hashes = in_ev + in_repo + not_fnd
        if verbose:
            print(
                f"[qa] Citations (LLM, 3-level): "
                f"{len(in_ev)} in evidence, "
                f"{len(in_repo)} exists in repo ({', '.join(in_repo) or 'none'}), "
                f"{len(not_fnd)} not found ({', '.join(not_fnd) or 'none'})"
            )
    elif verbose:
        print("[qa] Citation check: n/a (fallback builds citations from retrieved hits)")

    elapsed = time.time() - t_start
    if verbose:
        print(f"[qa] Time taken: {elapsed:.2f}s (via {backend_used})")
        if elapsed > 20.0:
            print(
                f"[qa] [SLOW QUERY >20s] embedding model load={t_model_load:.2f}s, "
                f"retrieval={t_retrieval:.2f}s, Gemini call={t_gemini:.2f}s"
            )

    return {
        "answer": answer,
        "model_used": backend_used,
        "cited_hashes": cited_hashes,
        "in_evidence": len(in_ev),
        "exists_in_repo": len(in_repo),
        "not_found": len(not_fnd),
        "top3_distances": top3_distances,
        "abstained_at_retrieval": False,
        "seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# plain_gemini: Answer with no retrieval (baseline comparison)
# ---------------------------------------------------------------------------

def plain_gemini(
    question: str,
    chroma_dir: str | Path = CHROMA_DIR,
) -> dict[str, Any]:
    """Call Gemini with no retrieval context (plain baseline).

    Plain English:
    Sends the question directly to Gemini with no evidence blocks — just a
    simple prompt. This is the "naive LLM" baseline for the benchmark.
    We then check whether any commit hashes it cites actually exist in
    ChromaDB (exist_in_repo) or are made up (hashes_missing).

    Keys returned:
        answer              - Gemini's raw answer (or error string)
        cited_hashes        - deduplicated hashes extracted from the answer
        hashes_exist_in_repo - hashes that exist in the ChromaDB index
        hashes_missing      - hashes not in the index (likely hallucinated)
        seconds             - elapsed time
        error               - None on success, error string on failure
    """
    t_start = time.time()

    raw_key = os.getenv("GEMINI_API_KEY", "")
    api_key = raw_key.strip("<> '\"") if raw_key else ""

    if not api_key:
        return {
            "answer": "",
            "cited_hashes": [],
            "hashes_exist_in_repo": [],
            "hashes_missing": [],
            "seconds": 0.0,
            "error": "GEMINI_API_KEY not found or empty in .env",
        }

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {
            "answer": "",
            "cited_hashes": [],
            "hashes_exist_in_repo": [],
            "hashes_missing": [],
            "seconds": 0.0,
            "error": "google-genai SDK not installed",
        }

    prompt = f"Answer this question about the psf/requests GitHub repository: {question}"
    error_str: str | None = None
    answer = ""

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=25000),
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        answer = (response.text or "").strip()
        if not answer:
            error_str = "Gemini returned an empty response"
    except Exception as exc:
        error_str = f"{type(exc).__name__}: {str(exc)[:200]}"

    # Extract and classify hashes from answer
    repo_hashes = _get_repo_commit_hashes(chroma_dir)
    cited_raw = _CITATION_RE.findall(answer)
    # Also scan for bare 7–40 char hex strings outside brackets
    bare_hex = re.findall(r"\b([0-9a-fA-F]{7,40})\b", answer)
    all_raw = cited_raw + bare_hex
    # Deduplicate
    unique_cited = list(dict.fromkeys(c.strip().lower() for c in all_raw))

    hashes_exist: list[str] = []
    hashes_missing: list[str] = []

    for h in unique_cited:
        if (
            h in repo_hashes
            or any(rh.startswith(h) for rh in repo_hashes)
            or any(h.startswith(rh) for rh in repo_hashes)
        ):
            hashes_exist.append(h)
        else:
            hashes_missing.append(h)

    elapsed = time.time() - t_start

    return {
        "answer": answer,
        "cited_hashes": unique_cited,
        "hashes_exist_in_repo": hashes_exist,
        "hashes_missing": hashes_missing,
        "seconds": round(elapsed, 2),
        "error": error_str,
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def ask(
    question: str,
    chroma_dir: str | Path = CHROMA_DIR,
    top_k: int = TOP_K_COMBINED,
    force_structured: bool = False,
    verbose: bool = True,
) -> str:
    """Thin wrapper around ask_detailed() that returns only the answer string.

    Plain English:
    Calls ask_detailed() to do all the work (retrieval, abstain check, Gemini,
    fallback, citation validation, timing), prints the same log lines to stdout
    when verbose=True, and returns just the answer text.
    All callers keep working without any changes.
    """
    result = ask_detailed(
        question,
        chroma_dir=chroma_dir,
        top_k=top_k,
        force_structured=force_structured,
        verbose=verbose,
    )
    return result["answer"]



# ---------------------------------------------------------------------------
# CLI & Demo Mode
# ---------------------------------------------------------------------------

def interactive(chroma_dir: str | Path = CHROMA_DIR) -> None:
    """Run an interactive Q&A loop in the terminal."""
    print("=" * 72)
    print("  Codebase Archaeologist - Q&A Mode")
    print(f"  Model: {GEMINI_MODEL} | Abstain threshold: {ABSTAIN_DISTANCE}")
    print("  Ask questions about why code was changed in psf/requests.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 72)

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = ask(question, chroma_dir=chroma_dir)
        print("\n" + "-" * 72)
        print(answer)
        print("-" * 72)


if __name__ == "__main__":
    demo_queries = [
        "Why was _encode_files changed?",
        "Why were hasattr checks added to the codebase?",
        "What problem did file-like proxy objects cause?",
        "Why was the Content-Length header handling changed?",
        "What was the JSONDecodeError deserialization bug?",
    ]

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        print("=" * 72)
        print("  Codebase Archaeologist - Q&A Demo (5 Benchmark Queries)")
        print(f"  ChromaDB: {CHROMA_DIR}")
        print(f"  Main LLM: {GEMINI_MODEL} | Abstain distance: {ABSTAIN_DISTANCE}")
        print("=" * 72)

        for idx, q in enumerate(demo_queries, 1):
            print(f"\n{'='*72}")
            print(f"DEMO QUESTION {idx}/{len(demo_queries)}: {q}")
            print(f"{'='*72}")
            ans = ask(q, verbose=True)
            print("\n" + "-" * 72)
            print(ans)
            print("-" * 72)

    elif len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ans = ask(question, verbose=True)
        print("\n" + "-" * 72)
        print(ans)
        print("-" * 72)

    else:
        interactive()
