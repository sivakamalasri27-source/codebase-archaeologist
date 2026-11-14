"""Vector indexing module for Codebase Archaeologist.

Codebase Archaeologist - Week 3 / Week 4:
- Builds two kinds of ChromaDB documents:
    "commit"   -- commit message + PR title/body + linked issue text.
    "function" -- function identity + source code history summary.
- Embeds with the free, local sentence-transformers model all-MiniLM-L6-v2.
- Persists to ./chroma_db so the collection survives between runs.
- Incremental with change detection: re-runs skip unchanged docs,
  delete+re-embed docs whose text changed, add brand-new docs.
- search(query, n) retrieves the top-n semantically similar documents.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lazy imports -- sentence-transformers and chromadb are heavy; keep at module
# level but we import them once inside _load_model() / _get_collection().
# ---------------------------------------------------------------------------
_model = None          # cached SentenceTransformer instance
_collection = None     # cached ChromaDB collection

# Same regex as in extract.py -- finds (#NNN) or bare #NNN in commit messages
_PR_REF_RE = re.compile(r"\(#(\d+)\)|(?<!\w)#(\d+)")
# Finds issue links in PR bodies: "Fixes #123", "Closes #456", etc.
_ISSUE_LINK_RE = re.compile(
    r"(?:fixes|closes|resolves|fix|close|resolve)\s+#(\d+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model():
    """Load (and cache) the all-MiniLM-L6-v2 embedding model.

    Plain English: This function downloads the model the first time (about
    80 MB, stored in ~/.cache/huggingface) and then keeps it in memory so we
    don't reload it every call.  all-MiniLM-L6-v2 produces 384-dimensional
    vectors and runs in CPU-only mode -- no GPU, no API key needed.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("[index] Loading embedding model (all-MiniLM-L6-v2) ...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[index] Model loaded.")
    return _model


def _get_collection(chroma_dir: str | Path):
    """Return (and cache) the persistent ChromaDB collection.

    Plain English: ChromaDB stores vectors on disk in `chroma_dir`.  On the
    first run it creates the directory and an empty collection called
    'codebase_archaeologist'.  On every subsequent run it re-opens the same
    collection, so old embeddings are still there -- enabling incremental
    indexing.
    """
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        _collection = client.get_or_create_collection(
            name="codebase_archaeologist",
            # cosine distance is better than L2 for normalised sentence embeddings
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of text strings -> list of float vectors.

    Plain English: We hand the texts to the in-memory SentenceTransformer
    model in one batch call.  Batching is important -- the model is a
    transformer, so encoding 100 texts at once is much faster than 100
    individual calls.  `convert_to_list` turns numpy arrays into plain Python
    lists so ChromaDB can serialise them.
    """
    model = _load_model()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------

def build_commit_documents(
    commits: list[dict[str, Any]],
    pr_cache: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Turn raw commit dicts (from extract.py) into indexable documents.

    Args:
        commits: List of commit dicts returned by extract_commits().
        pr_cache: Optional dict returned by fetch_pr_and_issue_text() or
                  fetch_all_pr_details().  Used to enrich commit documents
                  with PR title + body + linked issue text.

    Returns:
        List of dicts with keys:
        - doc_id   : unique document ID (string "commit:<hash>")
        - text     : the text to embed (commit msg + PR + issue context)
        - metadata : dict stored alongside the vector in ChromaDB

    Plain English: We combine the commit message with any matching PR
    description and linked issue body.  The richer the text, the better
    our semantic search will be -- "Fix #1234" alone is useless; with
    PR title/body and the original issue it explains what was actually
    broken and why the fix works.  We truncate PR bodies to 1500 chars
    and issue bodies to 500 chars to stay within MiniLM's 256-token
    window (the model silently truncates but shorter is safer).
    """
    # Build fast lookups: PR number (int) -> PR dict
    pr_by_number: dict[int, dict[str, Any]] = {}
    # Issue number (int) -> issue dict
    issue_by_number: dict[int, dict[str, Any]] = {}

    if pr_cache:
        for pr in pr_cache.get("pull_requests", []):
            if pr.get("number"):
                pr_by_number[pr["number"]] = pr
        for iss in pr_cache.get("issues", []):
            if iss.get("number"):
                issue_by_number[iss["number"]] = iss

    documents: list[dict[str, Any]] = []
    pr_text_count = 0  # how many commits got PR context

    for c in commits:
        msg = c.get("message", "") or ""

        # Find the first PR number referenced in the commit message
        pr_title = ""
        pr_body_snippet = ""
        issue_snippet = ""

        for m in _PR_REF_RE.finditer(msg):
            pr_num = int(m.group(1) or m.group(2))
            pr = pr_by_number.get(pr_num)
            if not pr:
                continue

            pr_title = pr.get("title", "")
            pr_body_snippet = (pr.get("body", "") or "")[:1500]

            # Look for linked issues inside the PR body
            full_body = pr.get("body", "") or ""
            for issue_m in _ISSUE_LINK_RE.finditer(full_body):
                iss_num = int(issue_m.group(1))
                iss = issue_by_number.get(iss_num)
                if iss:
                    iss_title = iss.get("title", "")
                    iss_body = (iss.get("body", "") or "")[:500]
                    issue_snippet += f"\nLinked issue #{iss_num}: {iss_title}\n{iss_body}"
            break  # only use the first matched PR per commit

        # Build the text we will embed
        parts = [f"Commit {c['hash'][:8]} by {c.get('author', '')}", msg]
        if pr_title:
            parts.append(f"PR: {pr_title}")
            pr_text_count += 1
        if pr_body_snippet:
            parts.append(pr_body_snippet)
        if issue_snippet:
            parts.append(issue_snippet.strip())

        text = "\n".join(parts).strip()

        # Metadata stored verbatim (not embedded) -- annotates search results
        changed_identities = [diff.get("path", "") for diff in c.get("python_diffs", [])]

        documents.append({
            "doc_id": f"commit:{c['hash']}",
            "text": text,
            "metadata": {
                "type": "commit",
                "hash": c["hash"],
                "short_hash": c["hash"][:8],
                "date": c.get("date", ""),
                "author": c.get("author", ""),
                "message": msg.splitlines()[0] if msg else "",
                # ChromaDB metadata values must be str/int/float/bool -- join lists
                "changed_files": ", ".join(changed_identities[:10]),
            },
        })

    total = len(commits)
    pct = 100.0 * pr_text_count / total if total else 0.0
    print(f"[build_commit_documents] Commits with PR text: "
          f"{pr_text_count}/{total} ({pct:.1f}%)")

    return documents


def build_function_documents(
    histories: dict[str, Any],  # dict[identity, FunctionHistory] from track.py
) -> list[dict[str, Any]]:
    """Turn function histories (from track.py) into indexable documents.

    Args:
        histories: The dict returned by build_function_histories().

    Returns:
        List of dicts with doc_id, text, metadata.

    Plain English: For each function we build a text that contains:
    1. Its identity (e.g. "models.py::PreparedRequest.prepare_body")
    2. A one-line history summary ("Changed 3 times. Last by Nate Prewitt:
       Add inline types to Requests")
    3. The commit messages of every commit that touched it.

    We do NOT embed the full source code -- MiniLM has a 256-token limit and
    source code doesn't describe *why* something changed.  The commit messages
    and PR text do.
    """
    documents: list[dict[str, Any]] = []
    for identity, hist in histories.items():
        # hist is a FunctionHistory dataclass; access its fields
        commits = hist.commits   # list[CommitRecord]
        aliases = hist.aliases   # list[str]

        if not commits:
            continue   # never actually changed -- skip

        last = commits[-1]
        # Summarise: who changed it, how often, and with what messages
        history_lines = [f"  - [{cr.short_hash}] {cr.date[:10]} {cr.author}: {cr.message}"
                         for cr in commits[-10:]]   # last 10 commits max

        alias_note = ""
        if aliases:
            alias_note = f"\nFormerly: {', '.join(aliases[-3:])}"

        text = (
            f"Function: {identity}{alias_note}\n"
            f"Changed {len(commits)} time(s). "
            f"Last modified by {last.author} on {last.date[:10]}: {last.message}\n"
            f"Change history:\n" + "\n".join(history_lines)
        ).strip()

        # Extract file path from identity (format: "path/to/file.py::Name")
        file_path = identity.split("::")[0] if "::" in identity else identity

        documents.append({
            "doc_id": f"function:{identity}",
            "text": text,
            "metadata": {
                "type": "function",
                "identity": identity,
                "file": file_path,
                "num_commits": len(commits),
                "last_author": last.author,
                "last_date": last.date[:10] if last.date else "",
                "last_message": last.message,
                "aliases": ", ".join(aliases) if aliases else "",
            },
        })
    return documents


# ---------------------------------------------------------------------------
# Core indexing
# ---------------------------------------------------------------------------

def index_documents(
    documents: list[dict[str, Any]],
    chroma_dir: str | Path = "./chroma_db",
) -> dict[str, int]:
    """Embed and store documents in ChromaDB with change detection.

    Args:
        documents: List of dicts with doc_id, text, metadata.
        chroma_dir: Path where ChromaDB persists its data.

    Returns:
        A dict {"added": N, "changed": C, "skipped": M} reporting what happened.

    Plain English: We first ask ChromaDB which doc_ids it already has AND
    what text they store.  Then we compare stored text vs. new text:
    - If a doc_id is brand new: embed and add it.
    - If the text changed (e.g. because we added PR context): delete the old
      embedding and re-embed with the new richer text.
    - If the text is identical: skip entirely (truly incremental).
    This means re-running after a PR-enrichment step will only re-embed the
    documents that actually gained new PR text, not all 596.
    """
    collection = _get_collection(chroma_dir)

    if not documents:
        return {"added": 0, "changed": 0, "skipped": 0}

    all_ids = [d["doc_id"] for d in documents]

    # Fetch stored text for every ID that already exists in ChromaDB
    stored_text_by_id: dict[str, str] = {}
    try:
        result = collection.get(ids=all_ids, include=["documents"])
        for doc_id, stored_text in zip(result["ids"], result["documents"]):
            stored_text_by_id[doc_id] = stored_text
    except Exception:
        pass  # collection empty or other issue -- treat everything as new

    new_docs: list[dict[str, Any]] = []
    changed_docs: list[dict[str, Any]] = []
    skipped = 0

    for d in documents:
        stored = stored_text_by_id.get(d["doc_id"])
        if stored is None:
            new_docs.append(d)
        elif stored != d["text"]:
            changed_docs.append(d)
        else:
            skipped += 1

    print(f"[index_documents] New: {len(new_docs)}, "
          f"Changed: {len(changed_docs)}, Skipped: {skipped}")

    # Delete the stale embeddings for changed documents before re-embedding
    if changed_docs:
        collection.delete(ids=[d["doc_id"] for d in changed_docs])

    # Embed and upsert new + changed documents together in batches of 100
    to_embed = new_docs + changed_docs
    added = 0
    batch_size = 100

    for start in range(0, len(to_embed), batch_size):
        batch = to_embed[start: start + batch_size]
        texts = [d["text"] for d in batch]
        embeddings = _embed(texts)

        collection.upsert(
            ids=[d["doc_id"] for d in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[d["metadata"] for d in batch],
        )
        added += len(batch)
        print(f"[index] Indexed {min(start + batch_size, len(to_embed))}/{len(to_embed)} docs ...")

    return {"added": len(new_docs), "changed": len(changed_docs), "skipped": skipped}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    query: str,
    n: int = 5,
    chroma_dir: str | Path = "./chroma_db",
    doc_type: str | None = None,
) -> list[dict[str, Any]]:
    """Embed a natural-language query and return the top-n most similar documents.

    Args:
        query: Natural-language question, e.g. "why was _encode_files changed".
        n: Number of results to return.
        chroma_dir: Path to the persistent ChromaDB directory.
        doc_type: Optional filter -- "commit", "function", or None (search all).

    Returns:
        List of result dicts with keys: doc_id, text, metadata, distance.

    Plain English: We embed the query using the same model that produced the
    stored vectors.  ChromaDB then does an approximate nearest-neighbour (ANN)
    search using the HNSW index it built during indexing.  Distance is cosine
    distance (0 = identical, 2 = opposite); lower is better.
    """
    collection = _get_collection(chroma_dir)

    query_embedding = _embed([query])[0]

    where_filter: dict | None = None
    if doc_type:
        where_filter = {"type": {"$eq": doc_type}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict[str, Any]] = []
    if results and results.get("ids") and results["ids"][0]:
        for doc_id, text, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "doc_id": doc_id,
                "text": text,
                "metadata": meta,
                "distance": round(dist, 4),
            })
    return hits


def _print_results(query: str, results: list[dict[str, Any]]) -> None:
    """Pretty-print search results to stdout."""
    print(f"\n{'='*72}")
    print(f"  SEARCH: \"{query}\"")
    print(f"{'='*72}")
    if not results:
        print("  No results found.")
        return
    for rank, hit in enumerate(results, 1):
        m = hit["metadata"]
        print(f"\n  [{rank}] dist={hit['distance']:.4f}  type={m.get('type','?')}")
        if m.get("type") == "commit":
            print(f"       {m.get('short_hash','')}  {m.get('date','')[:10]}  {m.get('author','')}")
            print(f"       {m.get('message','')}")
            if m.get("changed_files"):
                print(f"       Files: {m['changed_files'][:80]}")
        elif m.get("type") == "function":
            print(f"       {m.get('identity','')}")
            print(f"       Changed {m.get('num_commits',0)}x, last by {m.get('last_author','')} on {m.get('last_date','')}")
            print(f"       {m.get('last_message','')}")
        snippet = "\n".join(hit["text"].splitlines()[:2])
        print(f"       >> {snippet}")
    print(f"\n{'='*72}")


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

    from extract import extract_commits, fetch_all_pr_details
    from track import build_function_histories

    project_root = Path(__file__).resolve().parent.parent
    repo_path    = project_root / "repos" / "requests"
    chroma_dir   = project_root / "chroma_db"
    cache_path   = project_root / "cache_github_requests.json"

    if not repo_path.exists():
        print(f"Repo not found at {repo_path}. Please clone psf/requests there first.")
        sys.exit(1)

    total_start = time.time()

    # 1. Load commits
    print("\n[1/5] Extracting commits ...")
    t = time.time()
    commits = extract_commits(str(repo_path), max_commits=300)
    print(f"      {len(commits)} commits loaded in {time.time()-t:.1f}s")

    # 2. Fetch all PRs referenced in commit messages (uses cache; network only for missing ones)
    print("\n[2/5] Fetching PR details for all referenced PRs ...")
    t = time.time()
    pr_cache = fetch_all_pr_details(
        commits=commits,
        owner="psf",
        repo="requests",
        cache_path=cache_path,
    )
    print(f"      Done in {time.time()-t:.1f}s")

    # 3. Build function histories
    print("\n[3/5] Building function histories ...")
    t = time.time()
    histories = build_function_histories(str(repo_path), max_commits=300)
    print(f"      {len(histories)} function identities in {time.time()-t:.1f}s")

    # 4. Build documents
    print("\n[4/5] Building index documents ...")
    t = time.time()
    commit_docs   = build_commit_documents(commits, pr_cache)
    function_docs = build_function_documents(histories)
    all_docs      = commit_docs + function_docs
    print(f"      {len(commit_docs)} commit docs + {len(function_docs)} function docs "
          f"= {len(all_docs)} total ({time.time()-t:.1f}s)")

    # 5. Embed + store (with change detection)
    print(f"\n[5/5] Embedding and indexing into ChromaDB at {chroma_dir} ...")
    t = time.time()
    stats = index_documents(all_docs, chroma_dir=chroma_dir)
    embed_time = time.time() - t
    print(f"      Added {stats['added']} new, {stats['changed']} changed, "
          f"{stats['skipped']} skipped ({embed_time:.1f}s)")

    total_time = time.time() - total_start
    print(f"\n[index] Total time: {total_time:.1f}s")
    print(f"[index] ChromaDB collection size: {_get_collection(chroma_dir).count()} documents")

    # 6. Demo search
    _print_results(
        "why was _encode_files changed",
        search("why was _encode_files changed", n=5, chroma_dir=chroma_dir),
    )

    print("\n  Tip: Run again to test incremental indexing (all docs should be skipped).")
