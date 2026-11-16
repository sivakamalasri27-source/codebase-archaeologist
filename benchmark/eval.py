"""Benchmark evaluation runner for Codebase Archaeologist.

Plain English:
For each question in ground_truth.csv this script:
  1. Runs the full retrieval + LLM pipeline via ask_detailed().
  2. Runs a plain (no-retrieval) Gemini call via plain_gemini().
  3. Writes one row per question to benchmark/results.csv.
  4. NEVER sends the correct_answer column to any model.
  5. Leaves tool_correct, tool_gave_why, and plain_correct EMPTY — the user
     fills those in by hand after inspecting the CSV.

Retry policy: on 503 or 504 from plain_gemini(), wait 2 s and retry once.
If the second attempt also fails, record empty strings for the plain_* columns.
There is a 2 s sleep between every Gemini call to avoid rate-limit bursts.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — add src/ so we can import qa.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from qa import ask_detailed, plain_gemini  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCHMARK_CSV = Path(__file__).resolve().parent / "ground_truth.csv"
RESULTS_CSV   = Path(__file__).resolve().parent / "results.csv"

# Columns written to results.csv (last three are intentionally left empty)
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


def _join(lst: list[str]) -> str:
    """Join a list of strings with '|' for CSV storage."""
    return "|".join(lst) if lst else ""


def _call_plain_with_retry(question: str) -> dict:
    """Call plain_gemini() and retry once on 503/504 errors.

    Plain English:
    Calls the plain (no-retrieval) Gemini baseline. If the call returns an
    error string that mentions 503 or 504, waits 2 seconds and tries again.
    On second failure, returns a dict with empty answer so the row is still
    written to CSV.
    """
    result = plain_gemini(question)
    err = result.get("error") or ""
    if ("503" in err or "504" in err) and not result["answer"]:
        print(f"    [eval] plain_gemini got {err[:60]} — retrying in 2s…")
        time.sleep(2)
        result = plain_gemini(question)
    return result


def run_benchmark() -> None:
    print("=" * 80)
    print("  CODEBASE ARCHAEOLOGIST — BENCHMARK RUNNER")
    print(f"  Input : {BENCHMARK_CSV}")
    print(f"  Output: {RESULTS_CSV}")
    print("  Grading columns (tool_correct, tool_gave_why, plain_correct) are")
    print("  left EMPTY — fill them in by hand after reviewing the CSV.")
    print("=" * 80)

    with open(BENCHMARK_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []

    # Accumulators for the end-of-run summary
    abstain_test_count = 0
    total_tool_hallucinated = 0
    total_plain_hallucinated = 0

    for idx, row in enumerate(rows, 1):
        question   = row["question"]
        source_url = row.get("source_url", "")
        # NOTE: correct_answer is read ONLY to detect abstain tests.
        # It is never sent to any model.
        is_abstain_test = "NOT_IN_REPO" in row.get("correct_answer", "")

        if is_abstain_test:
            abstain_test_count += 1

        print(f"\n[{idx}/{len(rows)}] {'[ABSTAIN TEST] ' if is_abstain_test else ''}Q: {question}")

        # ------------------------------------------------------------------
        # Step 1: Run the tool (retrieval + Gemini / structured fallback)
        # ------------------------------------------------------------------
        print("    Running ask_detailed()…")
        tool = ask_detailed(question, verbose=False)
        total_tool_hallucinated += tool["not_found"]

        print(
            f"    tool: model={tool['model_used']}, "
            f"abstained={tool['abstained_at_retrieval']}, "
            f"not_found={tool['not_found']}, "
            f"seconds={tool['seconds']}"
        )

        # 2 s pause before Gemini call
        time.sleep(2)

        # ------------------------------------------------------------------
        # Step 2: Run plain Gemini (no retrieval, one retry on 503/504)
        # ------------------------------------------------------------------
        print("    Running plain_gemini()…")
        plain = _call_plain_with_retry(question)
        err = plain.get("error")
        if err:
            print(f"    plain_gemini error: {err[:80]}")
        total_plain_hallucinated += len(plain["hashes_missing"])

        print(
            f"    plain: hashes_exist={len(plain['hashes_exist_in_repo'])}, "
            f"hashes_missing={len(plain['hashes_missing'])}, "
            f"seconds={plain['seconds']}"
        )

        # 2 s pause before the next question's first Gemini call
        time.sleep(2)

        # ------------------------------------------------------------------
        # Step 3: Collect the result row
        # ------------------------------------------------------------------
        results.append({
            "question":             question,
            "source_url":           source_url,
            "is_abstain_test":      str(is_abstain_test),
            # Tool columns
            "tool_answer":          tool["answer"],
            "tool_hashes":          _join(tool["cited_hashes"]),
            "tool_in_evidence":     str(tool["in_evidence"]),
            "tool_exists_in_repo":  str(tool["exists_in_repo"]),
            "tool_not_found":       str(tool["not_found"]),
            "top3_distances":       _join([str(d) for d in tool["top3_distances"]]),
            "abstained_at_retrieval": str(tool["abstained_at_retrieval"]),
            "tool_seconds":         str(tool["seconds"]),
            # Plain Gemini columns
            "plain_answer":         plain["answer"],
            "plain_hashes":         _join(plain["cited_hashes"]),
            "plain_hashes_exist":   _join(plain["hashes_exist_in_repo"]),
            "plain_hashes_missing": _join(plain["hashes_missing"]),
            # Manual grading columns — intentionally empty
            "tool_correct":         "",
            "tool_gave_why":        "",
            "plain_correct":        "",
        })

    # -----------------------------------------------------------------------
    # Write CSV (utf-8-sig so Excel opens it without BOM issues)
    # -----------------------------------------------------------------------
    with open(RESULTS_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------------------------------------------
    # End-of-run summary (no PASS/FAIL grading)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"  Abstain-test questions          : {abstain_test_count}")
    print(f"  Total tool hallucinated hashes  : {total_tool_hallucinated}")
    print(f"  Total plain Gemini hallucinated : {total_plain_hallucinated}")
    print(f"  Results written to              : {RESULTS_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
