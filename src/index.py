"""Vector indexing module for Codebase Archaeologist.

Codebase Archaeologist - Week 3:
- Builds two kinds of ChromaDB documents:
    "commit"   — commit message + PR text, tagged with changed function identities.
    "function" — function identity + source code + history summary.
- Embeds with the free, local sentence-transformers model all-MiniLM-L6-v2.
- Persists to ./chroma_db so the collection survives between runs.
- Incremental: re-runs skip already-indexed documents (tracked by their document ID).
- search(query, n) retrieves the top-n semantically similar documents with metadata.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lazy imports — sentence-transformers and chromadb are heavy; keep at module
# level but we import them once inside _load_model() / _get_collection().
# ---------------------------------------------------------------------------
_model = None          # cached SentenceTransformer instance
_collection = None     # cached ChromaDB collection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model():
    """Load (and cache) the all-MiniLM-L6-v2 embedding model.

    Plain English: This function downloads the model the first time (about
    80 MB, stored in ~/.cache/huggingface) and then keeps it in memory so we
    don't reload it every call.  all-MiniLM-L6-v2 produces 384-dimensional
    vectors and runs in CPU-only mode — no GPU, no API key needed.
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
    collection, so old embeddings are still there — enabling incremental
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
    """Embed a list of text strings → list of float vectors.

    Plain English: We hand the texts to the in-memory SentenceTransformer
    model in one batch call.  Batching is important — the model is a
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
        pr_cache: Optional dict returned by fetch_pr_and_issue_text().
                  Used to enrich commit documents with PR title + body text.

    Returns:
        List of dicts with keys:
        - doc_id   : unique document ID (string "commit:<hash>")
        - text     : the text to embed
        - metadata : dict stored alongside the vector in ChromaDB

    Plain English: We combine the commit message with any PR description that
    matches via "#<number>" in the message.  The more context a document
    carries, the better our later semantic search will be — a bare "Fix #1234"
    is useless; with the PR body it explains what was actually broken.
    """
    # Build a fast lookup: PR number (int) -> PR dict
    pr_by_number: dict[int, dict[str, Any]] = {}
    if pr_cache:
        for pr in pr_cache.get("pull_requests", []):
            if pr.get("number"):
                pr_by_number[pr["number"]] = pr

    documents: list[dict[str, Any]] = []
    for c in commits:
        msg = c.get("message", "") or ""
        # Try to find a PR number reference like (#1234) or #1234
        pr_text = ""
        for token in msg.replace("(", " ").replace(")", " ").split():
            if token.startswith("#") and token[1:].isdigit():
                pr_num = int(token[1:])
                pr = pr_by_number.get(pr_num)
                if pr:
                    pr_text = f"\nPR: {pr.get('title', '')}\n{pr.get('body', '')[:800]}"
                break

        # The text we embed: commit message + optional PR context
        text = f"Commit {c['hash'][:8]} by {c.get('author', '')}\n{msg}{pr_text}".strip()

        # Metadata stored verbatim (not embedded) — used to annotate results
        changed_identities = []
        for diff in c.get("python_diffs", []):
            # Store the file path; full identities come from track.py histories
            changed_identities.append(diff.get("path", ""))

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
                # ChromaDB metadata values must be str/int/float/bool — join lists
                "changed_files": ", ".join(changed_identities[:10]),
            },
        })
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

    We do NOT embed the full source code — MiniLM has a 256-token limit and
    source code doesn't describe *why* something changed.  The commit messages
    and PR text do.
    """
    documents: list[dict[str, Any]] = []
    for identity, hist in histories.items():
        # hist is a FunctionHistory dataclass; access its fields
        commits = hist.commits   # list[CommitRecord]
        aliases = hist.aliases   # list[str]

        if not commits:
            continue   # never actually changed — skip

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
    """Embed and store documents in ChromaDB, skipping already-indexed ones.

    Args:
        documents: List of dicts with doc_id, text, metadata.
        chroma_dir: Path where ChromaDB persists its data.

    Returns:
        A dict {"added": N, "skipped": M} reporting what happened.

    Plain English: We first ask ChromaDB which doc_ids it already has.  Any
    document whose ID is already present is skipped — this is the incremental
    indexing mechanism.  The new documents are batched (100 at a time) so we
    don't hold huge numpy arrays in memory at once.  ChromaDB's `upsert` both
    adds new documents and updates existing ones if the same ID is re-submitted
    (but we avoid re-submitting existing IDs for speed).
    """
    collection = _get_collection(chroma_dir)

    # Find which document IDs already exist in the collection
    if documents:
        all_ids = [d["doc_id"] for d in documents]
        existing = set()
        # ChromaDB's get() can filter by IDs in one call
        try:
            result = collection.get(ids=all_ids, include=[])
            existing = set(result["ids"])
        except Exception:
            existing = set()

        new_docs = [d for d in documents if d["doc_id"] not in existing]
    else:
        new_docs = []

    skipped = len(documents) - len(new_docs)
    added = 0

    if not new_docs:
        return {"added": 0, "skipped": skipped}

    # Batch embed + upsert in chunks of 100
    batch_size = 100
    for start in range(0, len(new_docs), batch_size):
        batch = new_docs[start : start + batch_size]
        texts = [d["text"] for d in batch]
        embeddings = _embed(texts)

        collection.upsert(
            ids=[d["doc_id"] for d in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[d["metadata"] for d in batch],
        )
        added += len(batch)
        print(f"[index] Indexed {min(start + batch_size, len(new_docs))}/{len(new_docs)} new documents ...")

    return {"added": added, "skipped": skipped}


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
        doc_type: Optional filter — "commit", "function", or None (search all).

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


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from extract import extract_commits
    from parse import parse_definitions
    from track import build_function_histories

    project_root = Path(__file__).resolve().parent.parent
    repo_path    = project_root / "repos" / "requests"
    chroma_dir   = project_root / "chroma_db"
    cache_path   = project_root / "cache_github_requests.json"

    if not repo_path.exists():
        print(f"Repo not found at {repo_path}. Please clone psf/requests there first.")
        sys.exit(1)

    total_start = time.time()

    # 1. Load commits (extract step)
    print("\n[1/4] Extracting commits ...")
    t = time.time()
    commits = extract_commits(str(repo_path), max_commits=300)
    print(f"      {len(commits)} commits loaded in {time.time()-t:.1f}s")

    # 2. Load PR/issue cache for enrichment (may be empty if no network run yet)
    pr_cache: dict[str, Any] | None = None
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            pr_cache = json.load(f)
        n_prs = len((pr_cache or {}).get("pull_requests", []))
        print(f"      Loaded PR cache with {n_prs} PRs from {cache_path.name}")
    else:
        print(f"      No PR cache found at {cache_path}. Commit docs will not have PR text.")

    # 3. Build function histories (track step)
    print("\n[2/4] Building function histories ...")
    t = time.time()
    histories = build_function_histories(str(repo_path), max_commits=300)
    print(f"      {len(histories)} function identities in {time.time()-t:.1f}s")

    # 4. Build documents
    print("\n[3/4] Building index documents ...")
    t = time.time()
    commit_docs   = build_commit_documents(commits, pr_cache)
    function_docs = build_function_documents(histories)
    all_docs      = commit_docs + function_docs
    print(f"      {len(commit_docs)} commit docs + {len(function_docs)} function docs "
          f"= {len(all_docs)} total ({time.time()-t:.1f}s)")

    # 5. Embed + store
    print(f"\n[4/4] Embedding and indexing into ChromaDB at {chroma_dir} ...")
    t = time.time()
    stats = index_documents(all_docs, chroma_dir=chroma_dir)
    embed_time = time.time() - t
    print(f"      Added {stats['added']} new docs, skipped {stats['skipped']} already-indexed "
          f"({embed_time:.1f}s)")

    total_time = time.time() - total_start
    print(f"\n[index] Total time: {total_time:.1f}s")
    print(f"[index] ChromaDB collection size: {_get_collection(chroma_dir).count()} documents")

    # 6. Demo search
    query = "why was _encode_files changed"
    print(f"\n{'='*72}")
    print(f"  SEARCH: \"{query}\"")
    print(f"{'='*72}")

    results = search(query, n=5, chroma_dir=chroma_dir)
    if not results:
        print("  No results found.")
    else:
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
            # Print first 2 lines of the stored text for context
            snippet = "\n".join(hit["text"].splitlines()[:2])
            print(f"       >> {snippet}")

    print(f"\n{'='*72}")
    print("  Tip: Run again to test incremental indexing (all docs should be skipped).")
    print(f"{'='*72}\n")
