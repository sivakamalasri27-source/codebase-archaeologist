"""Function history tracker module.

Codebase Archaeologist - Week 2 (Part 2):
- Assigns every function/method/class a stable string identity: "file::QualifiedName".
- Walks commits oldest-to-newest and builds a per-identity change history.
- Detects renames and file moves using difflib similarity >= 0.8.
  When detected, the old identity's history is merged into the new identity.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydriller import Repository

from parse import parse_definitions, find_changed_functions_in_diff


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CommitRecord:
    """Lightweight snapshot of one commit that touched a function."""
    hash: str           # Full SHA-1 commit hash
    short_hash: str     # First 8 characters for display
    date: str           # ISO-8601 author date string
    author: str         # Author name
    message: str        # First line of the commit message


@dataclass
class FunctionHistory:
    """The accumulated change history for one stable function identity."""
    identity: str                          # e.g. "src/requests/models.py::Session.send"
    aliases: list[str] = field(default_factory=list)   # Previous identities before renames
    commits: list[CommitRecord] = field(default_factory=list)   # Ordered oldest → newest


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

def make_identity(file_path: str, qualified_name: str) -> str:
    """Build a stable identity key from a file path and a qualified name.

    Args:
        file_path: Relative path to the Python file (forward slashes).
        qualified_name: Qualified name e.g. 'Session.send' or 'check_header'.

    Returns:
        A string like 'src/requests/sessions.py::Session.send'.
    """
    # Normalise Windows backslashes
    normalised = file_path.replace("\\", "/")
    return f"{normalised}::{qualified_name}"


def _normalise_path(raw: str | None) -> str:
    """Return a forward-slash path, or empty string if None."""
    return (raw or "").replace("\\", "/")


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

def build_function_histories(
    repo_path: str,
    max_commits: int = 300,
    rename_threshold: float = 0.8,
) -> dict[str, FunctionHistory]:
    """Walk up to `max_commits` commits (oldest first) and build a history map.

    For each commit that modifies a Python file, the function:
    1. Parses the **before** and **after** versions of every changed .py file.
    2. Identifies which functions/methods were modified (using the diff).
    3. Detects renames/moves: a definition that disappears in one file and
       reappears (with ≥ 0.8 code similarity) anywhere in the same commit is
       treated as the same entity, and its histories are merged.
    4. Appends a CommitRecord to every identity touched in that commit.

    Args:
        repo_path: Local path to the git repository.
        max_commits: Maximum number of commits to inspect (most-recent N).
        rename_threshold: Minimum SequenceMatcher similarity to call a rename.

    Returns:
        dict mapping identity string → FunctionHistory, ordered oldest→newest.
    """
    repo_path = str(repo_path)

    # Step 1 – collect the N most-recent commits in reverse order, then flip
    # so we process oldest-first (histories build up chronologically).
    import time
    t0 = time.time()
    print(f"[track] Collecting up to {max_commits} commits ...")
    raw_commits: list[Any] = []
    for c in Repository(repo_path, order="reverse").traverse_commits():
        raw_commits.append(c)
        if len(raw_commits) >= max_commits:
            break
    raw_commits.reverse()   # now oldest to newest
    print(f"[track] Processing {len(raw_commits)} commits oldest to newest ...")

    histories: dict[str, FunctionHistory] = {}

    for i, commit in enumerate(raw_commits, 1):
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"[track] Progress: {i}/{len(raw_commits)} commits ({elapsed:.1f}s elapsed) ...")
        try:
            modified_files = commit.modified_files
        except Exception:
            # Shallow clone boundary – parent diff unavailable; skip safely.
            continue

        # Collect per-commit: what disappeared, what appeared, what changed.
        disappeared: dict[str, str] = {}  # identity -> source_code (before)
        appeared:    dict[str, str] = {}  # identity -> source_code (after)
        changed_ids: set[str]       = set()

        for mod in modified_files:
            new_path = _normalise_path(mod.new_path or mod.filename)
            old_path = _normalise_path(mod.old_path or mod.filename)

            if not new_path.endswith(".py") and not old_path.endswith(".py"):
                continue

            # --- Definitions before this commit ---
            defs_before: dict[str, str] = {}
            if mod.source_code_before:
                for d in parse_definitions(mod.source_code_before):
                    key = make_identity(old_path, d["name"])
                    defs_before[key] = d["source_code"]

            # --- Definitions after this commit ---
            defs_after: dict[str, str] = {}
            if mod.source_code:
                for d in parse_definitions(mod.source_code):
                    key = make_identity(new_path, d["name"])
                    defs_after[key] = d["source_code"]

            # Definitions present in before but gone in after → candidates for rename
            for k, code in defs_before.items():
                if k not in defs_after:
                    disappeared[k] = code

            # Definitions present in after but new → candidates for rename target
            for k, code in defs_after.items():
                if k not in defs_before:
                    appeared[k] = code

            # Which definitions were actually modified (line-overlap with diff)?
            if mod.source_code and mod.diff:
                for ch in find_changed_functions_in_diff(mod.source_code, mod.diff):
                    changed_ids.add(make_identity(new_path, ch["name"]))

        # -----------------------------------------------------------------
        # Rename / move detection (within this single commit)
        # -----------------------------------------------------------------
        rename_map: dict[str, str] = {}   # old_identity -> new_identity
        for old_id, old_code in list(disappeared.items()):
            best_score = 0.0
            best_new_id: str | None = None

            for new_id, new_code in list(appeared.items()):
                # Quick length filter: skip if too different in size
                len_a, len_b = len(old_code), len(new_code)
                if len_a == 0 or len_b == 0:
                    continue
                if min(len_a, len_b) / max(len_a, len_b) < 0.65:
                    continue

                sm = difflib.SequenceMatcher(None, old_code, new_code, autojunk=False)
                # Cheap upper bound first; only compute ratio() if promising
                if sm.real_quick_ratio() < rename_threshold:
                    continue
                if sm.quick_ratio() < rename_threshold:
                    continue
                score = sm.ratio()
                if score >= rename_threshold and score > best_score:
                    best_score = score
                    best_new_id = new_id

            if best_new_id is not None:
                rename_map[old_id] = best_new_id
                del appeared[best_new_id]   # consume so it won't match again

        # -----------------------------------------------------------------
        # Apply renames: merge old history into new identity
        # -----------------------------------------------------------------
        for old_id, new_id in rename_map.items():
            old_hist = histories.pop(old_id, None)
            if new_id not in histories:
                histories[new_id] = FunctionHistory(identity=new_id)
            if old_hist:
                # Prepend the old commits to the new identity's history
                histories[new_id].commits = old_hist.commits + histories[new_id].commits
                histories[new_id].aliases.extend(old_hist.aliases)
                histories[new_id].aliases.append(old_id)   # record the old name
            print(
                f"[rename] {old_id}\n"
                f"      -> {new_id}"
            )

        # -----------------------------------------------------------------
        # Record this commit in every identity that was changed
        # -----------------------------------------------------------------
        # Also include renamed identities (they were touched in this commit)
        all_touched = changed_ids | set(rename_map.values())

        cr = CommitRecord(
            hash=commit.hash,
            short_hash=commit.hash[:8],
            date=commit.author_date.isoformat() if commit.author_date else "",
            author=commit.author.name if commit.author else "Unknown",
            message=commit.msg.splitlines()[0].strip() if commit.msg else "",
        )

        for identity in all_touched:
            if identity not in histories:
                histories[identity] = FunctionHistory(identity=identity)
            # Avoid duplicate entries (same commit recorded twice)
            if not histories[identity].commits or histories[identity].commits[-1].hash != commit.hash:
                histories[identity].commits.append(cr)

    return histories


def get_function_history(
    histories: dict[str, FunctionHistory],
    identity: str,
) -> FunctionHistory | None:
    """Look up a function's full history by identity, with partial-match fallback.

    Args:
        histories: The map returned by build_function_histories().
        identity: Exact identity string, or a unique substring (e.g. 'Session.send').

    Returns:
        The matching FunctionHistory, or None if not found.
    """
    if identity in histories:
        return histories[identity]

    # Partial match: find all keys that contain the query as a substring
    matches = [h for key, h in histories.items() if identity in key]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer exact name component match
        exact = [h for h in matches if h.identity.split("::")[-1] == identity]
        if len(exact) == 1:
            return exact[0]
        # Return the one with the most commits (most prominent)
        return max(matches, key=lambda h: len(h.commits))
    return None


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------

def print_history(hist: FunctionHistory) -> None:
    """Print a function's complete commit history to stdout."""
    print(f"\n{'='*72}")
    print(f"  HISTORY: {hist.identity}")
    if hist.aliases:
        print(f"  Formerly known as:")
        for alias in hist.aliases:
            print(f"    • {alias}")
    print(f"  Total commits: {len(hist.commits)}")
    print(f"{'='*72}")
    if not hist.commits:
        print("  (No commits recorded in the analysed range)")
        return
    for i, cr in enumerate(hist.commits, 1):
        print(f"  [{i:>3}] {cr.short_hash}  {cr.date[:10]}  {cr.author}")
        print(f"        {cr.message}")
    print()


def print_renames(histories: dict[str, FunctionHistory], limit: int = 10) -> None:
    """Print up to `limit` detected renames/moves, sorted by number of aliases."""
    renamed = [h for h in histories.values() if h.aliases]
    renamed.sort(key=lambda h: len(h.commits), reverse=True)

    print(f"\n{'='*72}")
    print(f"  DETECTED RENAMES / MOVES  (showing up to {limit})")
    print(f"{'='*72}")
    if not renamed:
        print("  None detected in the analysed commit range.")
        return
    for hist in renamed[:limit]:
        for alias in hist.aliases:
            print(f"  {alias}")
            print(f"    -> {hist.identity}")
    print()


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parent.parent
    repo_path = project_root / "repos" / "requests"

    if not repo_path.exists() or not (repo_path / ".git").exists():
        print(f"Repo not found at {repo_path}. Please clone psf/requests there first.")
        sys.exit(1)

    # --- Build histories for 300 most-recent commits ---
    histories = build_function_histories(str(repo_path), max_commits=300)
    print(f"\n[track] Built histories for {len(histories)} unique identities.\n")

    # --- (a) Print full history for two specific functions ---
    targets = [
        "PreparedRequest.prepare_body",   # core HTTP body-building method
        "Session.send",                   # the main send pipeline
    ]
    for target in targets:
        hist = get_function_history(histories, target)
        if hist:
            print_history(hist)
        else:
            print(f"\n[warn] No history found for: {target!r}")
            # Show closest matches to help debug
            close = [k for k in histories if target.split(".")[-1] in k][:5]
            if close:
                print(f"  Closest keys: {close}")

    # --- (b) Print up to 10 detected renames / moves ---
    print_renames(histories, limit=10)
