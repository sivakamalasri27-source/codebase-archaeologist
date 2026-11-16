"""Print a readable summary of benchmark/results.csv.

Plain English:
Reads results.csv and, for each of the 7 rows, prints:
  - the question number and full question text
  - the full tool_answer (retrieval + LLM pipeline)
  - the full plain_answer (no-retrieval Gemini baseline)
  - abstained_at_retrieval, top3_distances, and tool_in_evidence count
A line of dashes separates each question.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"
DASH = "-" * 80


def show() -> None:
    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found. Run eval.py first.")
        sys.exit(1)

    with open(RESULTS_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for idx, row in enumerate(rows, 1):
        q           = row["question"]
        tool_ans    = row["tool_answer"]
        plain_ans   = row["plain_answer"]
        abstained   = row["abstained_at_retrieval"]
        distances   = row["top3_distances"]
        in_evidence = row["tool_in_evidence"]
        is_abstain  = row["is_abstain_test"]
        model_note  = "[ABSTAIN TEST]" if is_abstain == "True" else ""

        print(DASH)
        print(f"Q{idx} {model_note}  {q}")
        print(DASH)
        print(f"  abstained_at_retrieval : {abstained}")
        print(f"  top3_distances         : {distances}")
        print(f"  tool_in_evidence       : {in_evidence}")
        print()

        print("── TOOL ANSWER ──────────────────────────────────────────────────────────")
        if tool_ans.strip():
            for line in tool_ans.splitlines():
                print(f"  {line}")
        else:
            print("  (empty)")
        print()

        print("── PLAIN GEMINI ANSWER ──────────────────────────────────────────────────")
        if plain_ans.strip():
            for line in plain_ans.splitlines():
                print(f"  {line}")
        else:
            print("  (empty)")
        print()

    print(DASH)
    print(f"Total questions shown: {len(rows)}")
    print(DASH)


if __name__ == "__main__":
    show()
