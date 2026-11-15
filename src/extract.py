"""Git history extraction and GitHub API integration module.

Codebase Archaeologist:
- Extracts commit logs, file changes, and Python diffs using PyDriller.
- Fetches PR and Issue details via GitHub REST API (/issues/{number}).
- Caches API responses locally in JSON to prevent repeated requests.
- fetch_all_pr_details(): scans commit messages for #number references,
  fetches any missing items using the unified GitHub /issues/{number} endpoint,
  labels them as PRs or Issues, and crawls linked issues/PRs.
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

# Regex: match PR / issue references like "(#7502)" or "#7502" in commit messages
_PR_REF_RE = re.compile(r"\(#(\d+)\)|(?<!\w)#(\d+)")
# Regex: match issue links in PR/issue bodies like "Fixes #123", "Closes #456"
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
    """
    extracted_commits: list[dict[str, Any]] = []

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
    """Fetch initial PR descriptions and Issue discussions via GitHub REST API."""
    resolved_cache = Path(cache_path) if cache_path else Path(f"{owner}_{repo}_github_cache.json")

    if resolved_cache.exists():
        print(f"[CACHE HIT] Loading GitHub data from {resolved_cache.resolve()}")
        with open(resolved_cache, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[CACHE MISS] Fetching fresh data from GitHub REST API for {owner}/{repo}...")

    auth_token = token or os.getenv("GITHUB_TOKEN")
    headers = _build_headers(auth_token)
    base_url = f"https://api.github.com/repos/{owner}/{repo}"

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
    """Fetch every #number referenced in commits via GET /repos/{owner}/{repo}/issues/{number}.

    Plain English:
    GitHub shares one single numbering space between issues and PRs.
    Calling /pulls/{number} on an issue returns 404. Calling /issues/{number}
    works for both:
      - If 'pull_request' is in the payload, it is a PR (is_pr = True).
      - Otherwise, it is a regular issue (is_pr = False).
    We read GITHUB_TOKEN from .env, check what is already cached (never refetching
    existing items), fetch missing ones, and scan bodies for linked 'Fixes #N' items.
    """
    resolved_cache = Path(cache_path)
    auth_token = token or os.getenv("GITHUB_TOKEN")
    headers = _build_headers(auth_token)

    # 1. Load existing cache or initialize fresh
    if resolved_cache.exists():
        with open(resolved_cache, encoding="utf-8") as f:
            cache: dict[str, Any] = json.load(f)
    else:
        cache = {"repo": f"{owner}/{repo}", "pull_requests": [], "issues": []}

    # Ensure existing cache entries have is_pr flag and build fast lookup
    cached_by_num: dict[int, dict[str, Any]] = {}
    for pr in cache.setdefault("pull_requests", []):
        if pr.get("number"):
            pr.setdefault("is_pr", True)
            cached_by_num[pr["number"]] = pr

    for iss in cache.setdefault("issues", []):
        if iss.get("number"):
            iss.setdefault("is_pr", False)
            cached_by_num[iss["number"]] = iss

    # 2. Collect all referenced #numbers from commit messages
    referenced_nums: set[int] = set()
    for c in commits:
        msg = c.get("message", "") or ""
        for m in _PR_REF_RE.finditer(msg):
            num = int(m.group(1) or m.group(2))
            referenced_nums.add(num)

    missing_nums = sorted(referenced_nums - set(cached_by_num.keys()))
    print(f"[fetch_all_pr_details] {len(referenced_nums)} references in commits. "
          f"{len(cached_by_num)} already cached, {len(missing_nums)} to fetch.")

    failed_items: dict[int, tuple[int | None, str]] = {}
    newly_fetched = 0

    # Helper to fetch and cache a single issue/PR number
    def _fetch_and_cache(num: int) -> dict[str, Any] | None:
        nonlocal newly_fetched
        if num in cached_by_num:
            return cached_by_num[num]

        data, status_code, err_msg = _github_fetch_issue(owner, repo, num, headers)
        if data is None:
            failed_items[num] = (status_code, err_msg or "Unknown error")
            print(f"  [FAILURE] #{num}: HTTP {status_code} - {err_msg}")
            return None

        is_pr = "pull_request" in data
        item_record: dict[str, Any] = {
            "number": data.get("number", num),
            "title": data.get("title", ""),
            "body": data.get("body") or "",
            "author": (data.get("user") or {}).get("login", ""),
            "state": data.get("state", ""),
            "html_url": data.get("html_url", ""),
            "is_pr": is_pr,
            "created_at": data.get("created_at", ""),
        }

        if is_pr:
            cache["pull_requests"].append(item_record)
        else:
            cache["issues"].append(item_record)

        cached_by_num[num] = item_record
        newly_fetched += 1
        tag = "PR" if is_pr else "Issue"
        print(f"  [FETCHED] {tag} #{num}: {item_record['title'][:60]}")

        # Scan for linked issues in body (e.g. Fixes #123, Closes #456)
        body = item_record.get("body", "") or ""
        for linked_match in _ISSUE_LINK_RE.finditer(body):
            linked_num = int(linked_match.group(1))
            if linked_num not in cached_by_num:
                print(f"    -> Following linked item #{linked_num} from #{num}...")
                _fetch_and_cache(linked_num)

        _save_cache(resolved_cache, cache)
        return item_record

    # 3. Fetch each missing referenced number
    for num in missing_nums:
        _fetch_and_cache(num)

    # Attach tracking metrics to cache object
    cache["_referenced_nums"] = referenced_nums
    cache["_failed_items"] = failed_items

    print(f"[fetch_all_pr_details] Complete. Newly fetched: {newly_fetched}. "
          f"Cache now has {len(cache['pull_requests'])} PRs and {len(cache['issues'])} Issues.")

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


def _github_fetch_issue(
    owner: str,
    repo: str,
    number: int,
    headers: dict[str, str],
    max_retries: int = 3,
) -> tuple[dict[str, Any] | None, int | None, str | None]:
    """Fetch GET /repos/{owner}/{repo}/issues/{number} handling 403/429 retries.

    Returns:
        tuple of (data_dict_or_None, status_code_or_None, error_msg_or_None)
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
    delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            resp = http_requests.get(url, headers=headers, timeout=15)
        except http_requests.RequestException as err:
            return None, None, str(err)

        if resp.status_code == 200:
            return resp.json(), 200, None

        if resp.status_code in (403, 429):
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else delay
            print(f"  [RATE LIMIT] #{number} returned {resp.status_code}. "
                  f"Waiting {wait}s (retry {attempt}/{max_retries}) ...")
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue

        # Any non-rate-limit failure (e.g. 404, 401, 500)
        try:
            err_msg = resp.json().get("message", resp.text[:200])
        except Exception:
            err_msg = resp.reason or resp.text[:200]
        return None, resp.status_code, err_msg

    return None, 429, "Max rate limit retries exceeded"


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
        "is_pr": True,
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
        "is_pr": False,
    }


def _save_cache(cache_path: Path, data: dict[str, Any]) -> None:
    """Write cache data to disk as UTF-8 JSON (excluding private keys)."""
    try:
        clean_data = {k: v for k, v in data.items() if not k.startswith("_")}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)
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
    print("Codebase Archaeologist - Git Extraction Demo")
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

    print("\nTesting GitHub PR & Issue retrieval with unified /issues endpoint...")
    demo_cache_path = project_root / "cache_github_requests.json"
    data = fetch_all_pr_details(commits, "psf", "requests", cache_path=demo_cache_path)
    print("=" * 70)
