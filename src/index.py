"""Vector indexing module for Codebase Archaeologist.

Codebase Archaeologist:
- Builds two kinds of ChromaDB documents:
    "commit"   -- commit message + title + body of every referenced PR/issue.
    "function" -- function identity + change history summary.
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

# Regex: match PR / issue references in commit messages: (#123) or #123
_PR_REF_RE = re.compile(r"\(#(\d+)\)|(?<!\w)#(\d+)")
# Regex: match linked issue references in bodies: "Fixes #123", "Closes #456"
_ISSUE_LINK_RE = re.compile(
    r"(?:fixes|closes|resolves|fix|close|resolve)\s+#(\d+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model():
    """Load (and cache) the all-MiniLM-L6-v2 embedding model.

    Plain English: Downloads the model once (~80 MB) into ~/.cache/huggingface
    and caches it in memory. all-MiniLM-L6-v2 runs locally on CPU with no API key.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("[index] Loading embedding model (all-MiniLM-L6-v2) ...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[index] Model loaded.")
    return _model


def _get_collection(chroma_dir: str | Path):
    """Return (and cache) the persistent ChromaDB collection."""
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_dir))
        _collection = client.get_or_create_collection(
            name="codebase_archaeologist",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of text strings -> list of float vectors in batches."""
    model = _load_model()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------

def build_commit_documents(
    commits: list[dict[str, Any]],
    pr_cache: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Turn raw commit dicts into indexable documents enriched with PR/issue context.

    Args:
        commits: List of commit dicts from extract_commits().
        pr_cache: Dict returned by fetch_all_pr_details() containing pull_requests
                  and issues lists.

    Returns:
        tuple of (documents_list, commits_with_context_count)

    Plain English:
    For every commit message, find ALL referenced numbers (#NNN). For each
    number, look it up in the cache:
      - If it is a PR, mark it as 'PR #NNN: <title>' + body (max 1500 chars).
      - If it is an issue, mark it as 'Issue #NNN: <title>' + body (max 1500 chars).
      - Also attach any linked items ('Fixes #N') mentioned in their descriptions.
    """
    cached_by_number: dict[int, dict[str, Any]] = {}
    if pr_cache:
        for pr in pr_cache.get("pull_requests", []):
            if pr.get("number"):
                pr.setdefault("is_pr", True)
                cached_by_number[pr["number"]] = pr
        for iss in pr_cache.get("issues", []):
            if iss.get("number"):
                iss.setdefault("is_pr", False)
                cached_by_number[iss["number"]] = iss

    documents: list[dict[str, Any]] = []
    commits_with_context = 0

    for c in commits:
        msg = c.get("message", "") or ""

        # Extract all referenced numbers in order of appearance
        referenced_in_commit: list[int] = []
        for m in _PR_REF_RE.finditer(msg):
            num = int(m.group(1) or m.group(2))
            if num not in referenced_in_commit:
                referenced_in_commit.append(num)

        context_snippets: list[str] = []
        seen_linked: set[int] = set(referenced_in_commit)

        for num in referenced_in_commit:
            item = cached_by_number.get(num)
            if not item:
                continue

            tag = "PR" if item.get("is_pr") else "Issue"
            title = item.get("title", "").strip()
            body = (item.get("body", "") or "").strip()[:1500]

            snippet = f"{tag} #{num}: {title}\n{body}".strip()
            context_snippets.append(snippet)

            # Check for linked issues in body (e.g. Fixes #123)
            full_body = item.get("body", "") or ""
            for issue_m in _ISSUE_LINK_RE.finditer(full_body):
                linked_num = int(issue_m.group(1))
                if linked_num not in seen_linked:
                    seen_linked.add(linked_num)
                    linked_item = cached_by_number.get(linked_num)
                    if linked_item:
                        l_tag = "PR" if linked_item.get("is_pr") else "Issue"
                        l_title = linked_item.get("title", "").strip()
                        l_body = (linked_item.get("body", "") or "").strip()[:500]
                        context_snippets.append(
                            f"Linked {l_tag} #{linked_num}: {l_title}\n{l_body}".strip()
                        )

        parts = [f"Commit {c['hash'][:8]} by {c.get('author', '')}", msg]
        if context_snippets:
            commits_with_context += 1
            parts.extend(context_snippets)

        text = "\n\n".join(parts).strip()

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
                "changed_files": ", ".join(changed_identities[:10]),
            },
        })

    total = len(commits)
    pct = 100.0 * commits_with_context / total if total else 0.0
    print(f"[build_commit_documents] Commits with PR/issue text: {commits_with_context}/{total} ({pct:.1f}%)")

    return documents, commits_with_context


def build_function_documents(
    histories: dict[str, Any],
) -> list[dict[str, Any]]:
    """Turn function histories (from track.py) into indexable documents."""
    documents: list[dict[str, Any]] = []
    for identity, hist in histories.items():
        commits = hist.commits
        aliases = hist.aliases

        if not commits:
            continue

        last = commits[-1]
        history_lines = [
            f"  - [{cr.short_hash}] {cr.date[:10]} {cr.author}: {cr.message}"
            for cr in commits[-10:]
        ]

        alias_note = ""
        if aliases:
            alias_note = f"\nFormerly: {', '.join(aliases[-3:])}"

        text = (
            f"Function: {identity}{alias_note}\n"
            f"Changed {len(commits)} time(s). "
            f"Last modified by {last.author} on {last.date[:10]}: {last.message}\n"
            f"Change history:\n" + "\n".join(history_lines)
        ).strip()

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

    Plain English:
    Compares stored text vs new text in ChromaDB.
    - If doc is brand new: embeds and adds it.
    - If doc text changed: deletes stale vector and re-embeds the new text.
    - If doc text is identical: skips it entirely.
    """
    collection = _get_collection(chroma_dir)

    if not documents:
        return {"added": 0, "changed": 0, "skipped": 0}

    all_ids = [d["doc_id"] for d in documents]

    stored_text_by_id: dict[str, str] = {}
    try:
        result = collection.get(ids=all_ids, include=["documents"])
        for doc_id, stored_text in zip(result["ids"], result["documents"]):
            stored_text_by_id[doc_id] = stored_text
    except Exception:
        pass

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

    print(f"[index_documents] New: {len(new_docs)}, Changed: {len(changed_docs)}, Skipped: {skipped}")

    # Delete stale embeddings before re-embedding changed documents
    if changed_docs:
        collection.delete(ids=[d["doc_id"] for d in changed_docs])

    to_embed = new_docs + changed_docs
    added = 0
    batch_size = 100

    for start in range(0, len(to_embed), batch_size):
        batch = to_embed[start : start + batch_size]
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
    """Embed a query and return top-n nearest documents via cosine similarity."""
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

    from extract import extract_commits, fetch_all_pr_details, _PR_REF_RE
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

    # 2. Fetch all PRs/issues referenced in commit messages
    print("\n[2/5] Fetching PR & issue details via unified /issues endpoint ...")
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
    commit_docs, commits_with_context = build_commit_documents(commits, pr_cache)
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

    # 6. Calculate statistics for Requirement 7
    all_referenced_nums: set[int] = set()
    for c in commits:
        for m in _PR_REF_RE.finditer(c.get("message", "") or ""):
            all_referenced_nums.add(int(m.group(1) or m.group(2)))

    cached_lookup: dict[int, dict[str, Any]] = {}
    for pr in pr_cache.get("pull_requests", []):
        if pr.get("number"):
            cached_lookup[pr["number"]] = pr
    for iss in pr_cache.get("issues", []):
        if iss.get("number"):
            cached_lookup[iss["number"]] = iss

    pr_count = 0
    issue_count = 0
    for num in all_referenced_nums:
        item = cached_lookup.get(num)
        if item:
            if item.get("is_pr", False):
                pr_count += 1
            else:
                issue_count += 1

    failed_items = pr_cache.get("_failed_items", {})

    print("\n" + "=" * 72)
    print("FINAL SUMMARY REPORT")
    print("=" * 72)
    pct = 100.0 * commits_with_context / len(commits) if commits else 0.0
    print(f"a) Commits with PR/issue text  : {commits_with_context}/{len(commits)} ({pct:.1f}%)")
    print(f"b) Referenced numbers breakdown : {len(all_referenced_nums)} total ({pr_count} PRs, {issue_count} Issues)")
    if failed_items:
        print(f"c) Failed references ({len(failed_items)}):")
        for num, (code, reason) in sorted(failed_items.items()):
            print(f"   #{num}: HTTP {code} - {reason}")
    else:
        print("c) Failed references           : None (0 failed)")
    print("=" * 72)

    # 7. Demo search
    _print_results(
        "why was _encode_files changed",
        search("why was _encode_files changed", n=5, chroma_dir=chroma_dir),
    )
