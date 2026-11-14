"""Git history extraction and GitHub API integration module.

Week 1 / Week 4 Component of Codebase Archaeologist:
- Extracts commit logs, file changes, and Python diffs using PyDriller.
- Fetches PR descriptions and issue discussions using GitHub REST API.
- Caches API responses locally in JSON to prevent hitting rate limits.
- fetch_all_pr_details(): scans commit messages for (#NNN) references,
  fetches any missing PRs + linked issues from GitHub, updates cache.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydriller import Repository
import requests as http_requests

# Load environment variables from .env file if present
load_dotenv()

# Regex: match PR references like "(#7502)" or "#7502" in commit messages
_PR_REF_RE = re.compile(r"\(#(\d+)\)|(?<!\w)#(\d+)")
# Regex: match issue links in PR bodies like "Fixes #123", "Closes #456"
_ISSUE_LINK_RE = re.compile(
    r"(?:fixes|closes|resolves|fix|close|resolve)\s+#(\d+)",
    re.IGNORECASE,
)


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

    Plain English: We walk through every commit in the repo using PyDriller.
    For each commit we record who made it, when, what message they left, and
    what Python files changed along with their diff text.
    We stop at max_commits to keep indexing fast during development.
    """
    extracted_commits: list[dict[str, Any]] = []

    # Traverse commits in repository
    repo = Repository(repo_path, order=order)
    for commit in repo.traverse_commits():
        all_changed_files: list[str] = []
        python_diffs: list[dict[str, str]] = []

        try:
            modified_files = commit.modified_files
        except Exception:
            # Handles boundary commits in shallow clones where parents aren't present
            continue

        for modified_file in modified_files:
            file_path = (
                modified_file.new_path
                or modified_file.old_path
                or modified_file.filename
            )
            if file_path:
                all_changed_files.append(file_path)

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
        cache_path: Filepath for the JSON cache.
        per_page: Number of items to retrieve per API request (max 100).

    Returns:
        A dictionary containing repo, pull_requests, and issues lists.
    """
    resolved_cache = Path(cache_path) if cache_path else Path(f"{owner}_{repo}_github_cache.json")

    if resolved_cache.exists():
        print(f"[CACHE HIT] Loading GitHub data from {resolved_cache.resolve()}")
        with open(resolved_cache, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[CACHE MISS] Fetching fresh data from GitHub REST API for {owner}/{repo}...")

    auth_token = token or os.getenv("GITHUB_TOKEN")
    headers = _build_headers(auth_token)
    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    # 1. Fetch Pull Requests
    pull_requests: list[dict[str, Any]] = []
    try:
        pr_response = http_requests.get(
            f"{base_url}/pulls",
            headers=headers,
            params={"state": "all", "per_page": per_page, "sort": "updated", "direction": "desc"},
            timeout=15,
        )
        if pr_response.status_code == 200:
            for pr in pr_response.json():
                pull_requests.append(_normalise_pr(pr))
        else:
            print(f"[API ERROR] Pull Requests fetch returned status {pr_response.status_code}")
    except http_requests.RequestException as err:
        print(f"[REQUEST ERROR] Failed fetching pull requests: {err}")

    # 2. Fetch Issues (excluding PRs)
    issues: list[dict[str, Any]] = []
    try:
        issue_response = http_requests.get(
            f"{base_url}/issues",
            headers=headers,
            params={"state": "all", "per_page": per_page, "sort": "updated", "direction": "desc"},
            timeout=15,
        )
        if issue_response.status_code == 200:
            for item in issue_response.json():
                if "pull_request" in item:
                    continue
                issues.append(_normalise_issue(item))
        else:
            print(f"[API ERROR] Issues fetch returned status {issue_response.status_code}")
    except http_requests.RequestException as err:
        print(f"[REQUEST ERROR] Failed fetching issues: {err}")

    result_data: dict[str, Any] = {
        "repo": f"{owner}/{repo}",
        "pull_requests": pull_requests,
        "issues": issues,
    }

    _save_cache(resolved_cache, result_data)
    return result_data


def fetch_all_pr_details(
    commits: list[dict[str, Any]],
    owner: str,
    repo: str,
    cache_path: str | Path,
    token: str | None = None,
) -> dict[str, Any]:
    """Fetch every PR referenced in commit messages, plus any issues they link.

    Args:
        commits: List of commit dicts from extract_commits().
        owner: GitHub repo owner, e.g. 'psf'.
        repo: GitHub repo name, e.g. 'requests'.
        cache_path: Path to the JSON cache file (created if absent).
        token: GitHub PAT; falls back to GITHUB_TOKEN env var.

    Returns:
        Updated pr_cache dict with pull_requests and issues lists.

    Plain English:
    1. Scan every commit message for patterns like "(#7502)".
    2. For each PR number NOT yet in cache, fetch title and body from GitHub.
       If rate-limited (403/429), sleeps and retries. If unauthenticated limit
       is exhausted, safely halts remote calls and preserves cache.
    3. If PR body says "Fixes #123" or "Closes #123", fetch that issue too.
    4. Automatically writes updated cache to JSON so runs resume seamlessly.
    """
    resolved_cache = Path(cache_path)
    auth_token = token or os.getenv("GITHUB_TOKEN")
    headers = _build_headers(auth_token)
    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    # Load existing cache (or start fresh)
    if resolved_cache.exists():
        with open(resolved_cache, encoding="utf-8") as f:
            cache: dict[str, Any] = json.load(f)
    else:
        cache = {"repo": f"{owner}/{repo}", "pull_requests": [], "issues": []}

    cached_pr_nums: set[int] = {pr["number"] for pr in cache.get("pull_requests", []) if pr.get("number")}
    cached_issue_nums: set[int] = {iss["number"] for iss in cache.get("issues", []) if iss.get("number")}

    # Collect PR numbers referenced in commit messages
    referenced_pr_nums: set[int] = set()
    for c in commits:
        msg = c.get("message", "") or ""
        for m in _PR_REF_RE.finditer(msg):
            num = int(m.group(1) or m.group(2))
            referenced_pr_nums.add(num)

    missing_prs = sorted(referenced_pr_nums - cached_pr_nums)
    print(f"[fetch_all_pr_details] {len(referenced_pr_nums)} PR references in commits. "
          f"{len(cached_pr_nums)} already cached, {len(missing_prs)} to fetch.")

    newly_fetched_prs = 0
    newly_fetched_issues = 0

    for idx, pr_num in enumerate(missing_prs, 1):
        pr_data, hit_limit = _github_get(f"{base_url}/pulls/{pr_num}", headers)
        if hit_limit:
            print(f"[fetch_all_pr_details] Rate limit reached. Stopping further fetches.")
            break
        if pr_data is None:
            continue

        normalised = _normalise_pr(pr_data)
        cache.setdefault("pull_requests", []).append(normalised)
        cached_pr_nums.add(pr_num)
        newly_fetched_prs += 1
        print(f"  [{idx}/{len(missing_prs)}] PR #{pr_num}: {normalised['title'][:60]}")

        # Scan for linked issues
        body = normalised.get("body", "") or ""
        for issue_match in _ISSUE_LINK_RE.finditer(body):
            issue_num = int(issue_match.group(1))
            if issue_num in cached_issue_nums:
                continue
            issue_data, hit_limit_iss = _github_get(f"{base_url}/issues/{issue_num}", headers)
            if hit_limit_iss:
                break
            if issue_data and "pull_request" not in issue_data:
                cache.setdefault("issues", []).append(_normalise_issue(issue_data))
                cached_issue_nums.add(issue_num)
                newly_fetched_issues += 1
                print(f"    -> Linked Issue #{issue_num}: {issue_data.get('title','')[:50]}")

        # Persist after each PR so progress is never lost
        _save_cache(resolved_cache, cache)

    print(f"[fetch_all_pr_details] Done. Fetched {newly_fetched_prs} new PRs and "
          f"{newly_fetched_issues} new issues. Cache now has "
          f"{len(cache['pull_requests'])} PRs and {len(cache['issues'])} issues.")

    return cache


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_headers(auth_token: str | None) -> dict[str, str]:
    """Build standard GitHub API request headers."""
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if auth_token and auth_token.strip() and auth_token.strip() != "your_token_here":
        headers["Authorization"] = f"Bearer {auth_token.strip()}"
    else:
        print("[WARN] No valid GITHUB_TOKEN found. GitHub rate limit: 60 req/hr.")
    return headers


def _github_get(
    url: str,
    headers: dict[str, str],
    max_retries: int = 3,
) -> tuple[dict[str, Any] | None, bool]:
    """GET a single GitHub API URL with exponential back-off on 403/429.

    Returns:
        tuple (data_dict_or_None, rate_limit_hit_bool)
    """
    delay = 5
    for attempt in range(1, max_retries + 1):
        try:
            resp = http_requests.get(url, headers=headers, timeout=15)
        except http_requests.RequestException as err:
            print(f"  [HTTP ERROR] {url}: {err}")
            return None, False

        if resp.status_code == 200:
            return resp.json(), False

        if resp.status_code in (403, 429):
            rem = resp.headers.get("x-ratelimit-remaining")
            reset = resp.headers.get("x-ratelimit-reset")
            retry_after = resp.headers.get("Retry-After")

            if rem == "0" and reset:
                now = time.time()
                wait_sec = max(0, int(reset) - int(now)) + 1
                if wait_sec > 120 and "Authorization" not in headers:
                    print(f"  [RATE LIMIT] Unauthenticated GitHub rate limit exhausted (resets in {wait_sec}s).")
                    print("  [TIP] Provide a GITHUB_TOKEN in .env to enable 5,000 requests/hr.")
                    return None, True

            wait = int(retry_after) if retry_after and retry_after.isdigit() else delay
            print(f"  [RATE LIMIT] {resp.status_code} on {url}. Waiting {wait}s (retry {attempt}/{max_retries}) ...")
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code == 404:
            return None, False

        print(f"  [API ERROR] {resp.status_code} for {url}: {resp.text[:200]}")
        return None, False

    return None, True


def _normalise_pr(pr: dict[str, Any]) -> dict[str, Any]:
    """Extract standard fields from a raw GitHub PR object."""
    return {
        "number": pr.get("number"),
        "title": pr.get("title", ""),
        "body": pr.get("body") or "",
        "author": (pr.get("user") or {}).get("login", ""),
        "state": pr.get("state", ""),
        "created_at": pr.get("created_at", ""),
        "merged_at": pr.get("merged_at"),
        "html_url": pr.get("html_url", ""),
    }


def _normalise_issue(item: dict[str, Any]) -> dict[str, Any]:
    """Extract standard fields from a raw GitHub issue object."""
    return {
        "number": item.get("number"),
        "title": item.get("title", ""),
        "body": item.get("body") or "",
        "author": (item.get("user") or {}).get("login", ""),
        "state": item.get("state", ""),
        "comments_count": item.get("comments", 0),
        "created_at": item.get("created_at", ""),
        "html_url": item.get("html_url", ""),
    }


def _save_cache(cache_path: Path, data: dict[str, Any]) -> None:
    """Write cache data to disk as UTF-8 JSON."""
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as err:
        print(f"[CACHE WRITE ERROR] {err}")


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
    print("Codebase Archaeologist - Git Extraction Demo (Week 1 / Week 4)")
    print("=" * 70)

    project_root = Path(__file__).resolve().parent.parent
    sample_repo_path = project_root / "repos" / "requests"

    try:
        _ensure_sample_repo(sample_repo_path)
    except Exception as exc:
        print(f"[SETUP NOTICE] Could not clone automatically ({exc}).")
        sample_repo_path = Path(".")

    print(f"\nExtracting first 10 commits from {sample_repo_path} using PyDriller...")
    commits = extract_commits(str(sample_repo_path), max_commits=10)

    print(f"\nSuccessfully extracted {len(commits)} commits:\n")
    for idx, c in enumerate(commits, start=1):
        short_hash = c["hash"][:8]
        first_line_msg = c["message"].splitlines()[0] if c["message"] else "(no message)"
        print(f"[{idx:02d}] {short_hash} | {c['date'][:10]} | {c['author']}")
        print(f"     Msg: {first_line_msg}")
        print(f"     Changed files ({len(c['changed_files'])}): {', '.join(c['changed_files'][:4])}")

    print("\nTesting GitHub PR & Issue retrieval with local cache...")
    demo_cache_path = project_root / "cache_github_requests.json"
    data = fetch_all_pr_details(commits, "psf", "requests", cache_path=demo_cache_path)
    print("=" * 70)
