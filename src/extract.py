"""Git history extraction and GitHub API integration module.

Week 1 Component of Codebase Archaeologist:
- Extracts commit logs, file changes, and Python diffs using PyDriller.
- Fetches PR descriptions and issue discussions using GitHub REST API.
- Caches API responses locally to prevent rate limiting.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydriller import Repository
import requests

# Load environment variables from .env file if present
load_dotenv()


def extract_commits(
    repo_path: str,
    max_commits: int | None = None,
    order: str = "reverse",
) -> list[dict[str, Any]]:
    """Extract commit metadata and Python file diffs from a local git repository.

    Args:
        repo_path: Path to the local git repository directory.
        max_commits: Optional limit on the number of commits to extract.
        order: Traversal order ('reverse' for newest-to-oldest, None for chronological).

    Returns:
        A list of dictionaries where each dict represents a commit with:
        - hash (str): Full commit SHA hash.
        - author (str): Author name.
        - date (str): ISO-formatted commit author date.
        - message (str): Full commit message.
        - changed_files (list[str]): List of all modified/added/deleted file paths.
        - python_diffs (list[dict[str, str]]): List of diffs for modified .py files,
          each containing 'path' and 'diff' text.
    """
    extracted_commits: list[dict[str, Any]] = []

    # Traverse commits in repository
    repo = Repository(repo_path, order=order)
    for commit in repo.traverse_commits():
        # Collect all changed files across this commit
        all_changed_files: list[str] = []
        # Collect diffs only for Python source files
        python_diffs: list[dict[str, str]] = []

        try:
            modified_files = commit.modified_files
        except Exception:
            # Handles boundary commits in shallow clones where parents aren't present
            continue

        for modified_file in modified_files:
            # Resolve relative file path (prefer new_path for additions/modifications)
            file_path = (
                modified_file.new_path
                or modified_file.old_path
                or modified_file.filename
            )
            if file_path:
                all_changed_files.append(file_path)

            # Check if modified file is a Python script
            is_python_file = (
                (modified_file.filename and modified_file.filename.endswith(".py"))
                or (file_path and file_path.endswith(".py"))
            )
            if is_python_file:
                python_diffs.append({
                    "path": file_path or "unknown.py",
                    "diff": modified_file.diff or "",
                })

        commit_record: dict[str, Any] = {
            "hash": commit.hash,
            "author": commit.author.name if commit.author else "Unknown",
            "date": commit.author_date.isoformat() if commit.author_date else "",
            "message": commit.msg.strip(),
            "changed_files": all_changed_files,
            "python_diffs": python_diffs,
        }
        extracted_commits.append(commit_record)

        if max_commits is not None and len(extracted_commits) >= max_commits:
            break

    return extracted_commits


def fetch_pr_and_issue_text(
    owner: str,
    repo: str,
    token: str | None = None,
    cache_path: str | Path | None = None,
    per_page: int = 15,
) -> dict[str, Any]:
    """Fetch Pull Request descriptions and Issue discussions via GitHub REST API with JSON caching.

    Args:
        owner: GitHub repository owner (e.g., 'psf').
        repo: GitHub repository name (e.g., 'requests').
        token: GitHub Personal Access Token (defaults to GITHUB_TOKEN in .env).
        cache_path: Filepath for the JSON cache. Defaults to '{owner}_{repo}_github_cache.json'.
        per_page: Number of items to retrieve per API request (max 100).

    Returns:
        A dictionary containing:
        - repo: Full repository name ('owner/repo')
        - pull_requests: List of PR dicts (number, title, body, author, state, url)
        - issues: List of Issue dicts (number, title, body, author, state, comments_count, url)
    """
    # Default cache path in current working directory
    resolved_cache = Path(cache_path) if cache_path else Path(f"{owner}_{repo}_github_cache.json")

    # If cache exists, return cached data to avoid rate limits
    if resolved_cache.exists():
        print(f"[CACHE HIT] Loading GitHub data from {resolved_cache.resolve()}")
        with open(resolved_cache, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[CACHE MISS] Fetching fresh data from GitHub REST API for {owner}/{repo}...")

    # Resolve authorization token
    auth_token = token or os.getenv("GITHUB_TOKEN")
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    else:
        print("[WARN] No GITHUB_TOKEN found. Requests will be limited to unauthenticated quota (60 req/hr).")

    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    # 1. Fetch Pull Requests
    pull_requests: list[dict[str, Any]] = []
    try:
        pr_response = requests.get(
            f"{base_url}/pulls",
            headers=headers,
            params={"state": "all", "per_page": per_page, "sort": "updated", "direction": "desc"},
            timeout=15,
        )
        if pr_response.status_code == 200:
            for pr in pr_response.json():
                pull_requests.append({
                    "number": pr.get("number"),
                    "title": pr.get("title", ""),
                    "body": pr.get("body") or "",
                    "author": pr.get("user", {}).get("login", "") if pr.get("user") else "",
                    "state": pr.get("state", ""),
                    "created_at": pr.get("created_at", ""),
                    "merged_at": pr.get("merged_at"),
                    "html_url": pr.get("html_url", ""),
                })
        else:
            print(f"[API ERROR] Pull Requests fetch returned status {pr_response.status_code}: {pr_response.text[:200]}")
    except requests.RequestException as err:
        print(f"[REQUEST ERROR] Failed fetching pull requests: {err}")

    # 2. Fetch Issues (excluding PRs, as GitHub API returns PRs in issues endpoint)
    issues: list[dict[str, Any]] = []
    try:
        issue_response = requests.get(
            f"{base_url}/issues",
            headers=headers,
            params={"state": "all", "per_page": per_page, "sort": "updated", "direction": "desc"},
            timeout=15,
        )
        if issue_response.status_code == 200:
            for item in issue_response.json():
                # GitHub issues endpoint includes pull requests; check for 'pull_request' key
                if "pull_request" in item:
                    continue

                issues.append({
                    "number": item.get("number"),
                    "title": item.get("title", ""),
                    "body": item.get("body") or "",
                    "author": item.get("user", {}).get("login", "") if item.get("user") else "",
                    "state": item.get("state", ""),
                    "comments_count": item.get("comments", 0),
                    "created_at": item.get("created_at", ""),
                    "html_url": item.get("html_url", ""),
                })
        else:
            print(f"[API ERROR] Issues fetch returned status {issue_response.status_code}: {issue_response.text[:200]}")
    except requests.RequestException as err:
        print(f"[REQUEST ERROR] Failed fetching issues: {err}")

    result_data: dict[str, Any] = {
        "repo": f"{owner}/{repo}",
        "pull_requests": pull_requests,
        "issues": issues,
    }

    # Save to local JSON cache
    try:
        with open(resolved_cache, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"[CACHE SAVED] Saved {len(pull_requests)} PRs and {len(issues)} issues to {resolved_cache.resolve()}")
    except OSError as err:
        print(f"[CACHE WRITE ERROR] Failed saving cache file: {err}")

    return result_data


def _ensure_sample_repo(target_dir: Path) -> Path:
    """Clone a shallow copy of psf/requests if not already present locally."""
    if target_dir.exists() and (target_dir / ".git").exists():
        return target_dir

    print(f"Sample repo not found at {target_dir}. Cloning shallow copy of psf/requests...")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "20", "https://github.com/psf/requests.git", str(target_dir)],
        check=True,
    )
    return target_dir


if __name__ == "__main__":
    print("=" * 70)
    print("Codebase Archaeologist - Git Extraction Demo (Week 1)")
    print("=" * 70)

    # 1. Prepare local clone of psf/requests for testing
    project_root = Path(__file__).resolve().parent.parent
    sample_repo_path = project_root / "repos" / "requests"

    try:
        _ensure_sample_repo(sample_repo_path)
    except Exception as exc:
        print(f"[SETUP NOTICE] Could not clone automatically ({exc}).")
        print(f"Please clone psf/requests into: {sample_repo_path.resolve()}")
        sample_repo_path = Path(".")

    # 2. Extract first 10 commits
    print(f"\nExtracting first 10 commits from {sample_repo_path} using PyDriller...")
    commits = extract_commits(str(sample_repo_path), max_commits=10)

    print(f"\nSuccessfully extracted {len(commits)} commits:\n")
    for idx, c in enumerate(commits, start=1):
        short_hash = c["hash"][:8]
        first_line_msg = c["message"].splitlines()[0] if c["message"] else "(no message)"
        print(f"[{idx:02d}] {short_hash} | {c['date'][:10]} | {c['author']}")
        print(f"     Msg: {first_line_msg}")
        print(f"     Changed files ({len(c['changed_files'])}): {', '.join(c['changed_files'][:4])}")
        if len(c["changed_files"]) > 4:
            print(f"     ... + {len(c['changed_files']) - 4} more")

        print(f"     Python diffs ({len(c['python_diffs'])}):")
        for pydiff in c["python_diffs"][:2]:
            lines = pydiff["diff"].splitlines()
            snippet = "\n".join(lines[:3]) if lines else "(empty diff)"
            print(f"       * {pydiff['path']}:\n         {snippet.replace(chr(10), chr(10) + '         ')}")
        print("-" * 70)

    # 3. Demonstrate PR and Issue fetcher with caching
    print("\nTesting GitHub PR & Issue retrieval with local cache...")
    demo_cache_path = project_root / "cache_github_requests.json"
    data = fetch_pr_and_issue_text("psf", "requests", cache_path=demo_cache_path, per_page=5)
    print(f"Fetched {len(data.get('pull_requests', []))} PRs and {len(data.get('issues', []))} Issues.")
    if data.get("pull_requests"):
        first_pr = data["pull_requests"][0]
        print(f"Sample PR #{first_pr['number']}: {first_pr['title']} (by @{first_pr['author']})")
    print("=" * 70)
