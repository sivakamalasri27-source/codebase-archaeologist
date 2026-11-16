"""Re-run a single benchmark question and update only that row in results.csv.

Plain English:
Takes a question number (1–7) as a command-line argument.
Reads benchmark/results.csv, runs ask_detailed() and plain_gemini() for ONLY
that one question, and writes the updated row back to results.csv.

Crucially it PRESERVES your hand-filled columns:
  tool_correct, tool_gave_why, plain_correct
Everything else in that row is overwritten with the fresh run.

Retry/backup-model logic (same as eval.py):
  - On 503 or 504 from plain_gemini(), wait 2 s and retry once.
  - On second failure, keep the old plain_* values from the CSV
    (so you don't lose data if the API is overloaded).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qa import ask_detailed, plain_gemini  # noqa: E402

RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"

FIELDNAMES = [
    "question",
    "source_url",
    "is_abstain_test",
    "tool_answer",
    "tool_hashes",
    "tool_in_evidence",
    "tool_exists_in_repo",
    "tool_not_found",
    "top3_distances",
    "abstained_at_retrieval",
    "tool_seconds",
    "plain_answer",
    "plain_hashes",
    "plain_hashes_exist",
    "plain_hashes_missing",
    "tool_correct",
    "tool_gave_why",
    "plain_correct",
]

# Manual grading columns that must never be overwritten
PRESERVE = {"tool_correct", "tool_gave_why", "plain_correct"}


def _join(lst: list[str]) -> str:
    return "|".join(lst) if lst else ""


def _call_plain_with_retry(question: str) -> dict:
    """Call plain_gemini() with one retry on 503/504."""
    result = plain_gemini(question)
    err = result.get("error") or ""
    if ("503" in err or "504" in err) and not result["answer"]:
        print(f"  [rerun] plain_gemini got {err[:60]} — retrying in 2 s…")
        time.sleep(2)
        result = plain_gemini(question)
    return result


def rerun(question_number: int) -> None:
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run eval.py first.")
        sys.exit(1)

    # Read the existing CSV
    with open(RESULTS_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    n_rows = len(rows)
    if question_number < 1 or question_number > n_rows:
        print(f"ERROR: question number must be between 1 and {n_rows}.")
        sys.exit(1)

    row_idx = question_number - 1
    row = rows[row_idx]
    question = row["question"]

    print(f"\nRe-running Q{question_number}: {question}\n")

    # Preserve manual grading values BEFORE overwriting the row
    saved_manual = {col: row.get(col, "") for col in PRESERVE}

    # ------------------------------------------------------------------
    # Step 1: Run the tool pipeline
    # ------------------------------------------------------------------
    print("  Running ask_detailed()…")
    tool = ask_detailed(question, verbose=True)
    print(
        f"  tool: model={tool['model_used']}, "
        f"abstained={tool['abstained_at_retrieval']}, "
        f"not_found={tool['not_found']}, "
        f"seconds={tool['seconds']}"
    )

    time.sleep(2)

    # ------------------------------------------------------------------
    # Step 2: Run plain Gemini (with retry)
    # ------------------------------------------------------------------
    print("\n  Running plain_gemini()…")
    plain = _call_plain_with_retry(question)
    err = plain.get("error")
    if err:
        print(f"  plain_gemini error: {err[:80]}")
        # On persistent failure, keep old plain values so no data is lost
        if not plain["answer"]:
            print("  Keeping previous plain_* values from CSV.")
            plain_row_data = {
                "plain_answer":         row.get("plain_answer", ""),
                "plain_hashes":         row.get("plain_hashes", ""),
                "plain_hashes_exist":   row.get("plain_hashes_exist", ""),
                "plain_hashes_missing": row.get("plain_hashes_missing", ""),
            }
        else:
            plain_row_data = _build_plain_cols(plain)
    else:
        plain_row_data = _build_plain_cols(plain)

    print(
        f"  plain: hashes_exist={len(plain['hashes_exist_in_repo'])}, "
        f"hashes_missing={len(plain['hashes_missing'])}, "
        f"seconds={plain['seconds']}"
    )

    # ------------------------------------------------------------------
    # Step 3: Build the updated row
    # ------------------------------------------------------------------
    updated: dict[str, str] = {
        "question":             question,
        "source_url":           row.get("source_url", ""),
        "is_abstain_test":      row.get("is_abstain_test", ""),
        # Tool columns (freshly computed)
        "tool_answer":          tool["answer"],
        "tool_hashes":          _join(tool["cited_hashes"]),
        "tool_in_evidence":     str(tool["in_evidence"]),
        "tool_exists_in_repo":  str(tool["exists_in_repo"]),
        "tool_not_found":       str(tool["not_found"]),
        "top3_distances":       _join([str(d) for d in tool["top3_distances"]]),
        "abstained_at_retrieval": str(tool["abstained_at_retrieval"]),
        "tool_seconds":         str(tool["seconds"]),
        # Plain Gemini columns
        **plain_row_data,
        # Restored manual grading columns (never overwritten)
        **saved_manual,
    }

    rows[row_idx] = updated

    # ------------------------------------------------------------------
    # Step 4: Write the full CSV back (preserving all other rows)
    # ------------------------------------------------------------------
    with open(RESULTS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  ✓ Row {question_number} updated in {RESULTS_CSV}")
    print(f"  Preserved manual columns: {saved_manual}")


def _build_plain_cols(plain: dict) -> dict[str, str]:
    """Format plain_gemini() result into CSV-ready strings."""
    return {
        "plain_answer":         plain["answer"],
        "plain_hashes":         _join(plain["cited_hashes"]),
        "plain_hashes_exist":   _join(plain["hashes_exist_in_repo"]),
        "plain_hashes_missing": _join(plain["hashes_missing"]),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -X utf8 benchmark/rerun_one.py <question_number>")
        print("Example: python -X utf8 benchmark/rerun_one.py 7")
        sys.exit(1)

    try:
        q_num = int(sys.argv[1])
    except ValueError:
        print(f"ERROR: '{sys.argv[1]}' is not a valid integer.")
        sys.exit(1)

    rerun(q_num)
