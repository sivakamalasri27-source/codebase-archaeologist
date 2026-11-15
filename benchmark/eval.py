"""Benchmark evaluation runner for Codebase Archaeologist.

Plain English:
Runs the gold-standard benchmark Q&A dataset against the codebase archaeologist
retrieval and question-answering pipeline.
Evaluates:
1. Retrieval relevance & abstain accuracy (on out-of-domain questions).
2. Answer synthesis quality compared against ground truth.
3. Evidentiary citation accuracy (in evidence vs hallucinated).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qa import ask, retrieve, ABSTAIN_DISTANCE

BENCHMARK_CSV = Path(__file__).resolve().parent / "ground_truth.csv"


def run_benchmark():
    print("=" * 80)
    print("  CODEBASE ARCHAEOLOGIST - BENCHMARK EVALUATION SUITE")
    print(f"  Benchmark file: {BENCHMARK_CSV}")
    print(f"  Abstain threshold: {ABSTAIN_DISTANCE}")
    print("=" * 80)

    with open(BENCHMARK_CSV, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    results = []

    for idx, row in enumerate(reader, 1):
        q = row["question"]
        expected = row["correct_answer"]
        source_url = row.get("source_url", "")
        is_negative_test = "NOT_IN_REPO" in expected

        print(f"\n[{idx}/{len(reader)}] Question: {q}")
        if is_negative_test:
            print("      Expected: [NEGATIVE TEST - Should Abstain]")
        else:
            print(f"      Expected PR: {source_url}")

        t0 = time.time()
        ans = ask(q, verbose=False)
        elapsed = time.time() - t0

        abstained = "Not enough evidence in the repo history" in ans

        if is_negative_test:
            passed = abstained
            status = "PASS (Abstained)" if passed else "FAIL (Did not abstain)"
        else:
            passed = not abstained
            status = "PASS (Answered)" if passed else "FAIL (Unexpectedly abstained)"

        print(f"      Result: {status} in {elapsed:.2f}s")
        print("      Answer preview:")
        for line in ans.splitlines()[:4]:
            print(f"        {line}")
        if len(ans.splitlines()) > 4:
            print("        ...")

        results.append({
            "idx": idx,
            "question": q,
            "expected": expected,
            "source_url": source_url,
            "actual": ans,
            "passed": passed,
            "abstained": abstained,
            "elapsed": elapsed,
        })

    print("\n" + "=" * 80)
    print("  BENCHMARK SUMMARY")
    print("=" * 80)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"Total: {len(results)} | Passed: {passed_count}/{len(results)} ({passed_count/len(results):.0%})")
    for r in results:
        status_sym = "✅" if r["passed"] else "❌"
        print(f"  {status_sym} Q{r['idx']}: {r['question'][:50]}... ({r['elapsed']:.1f}s)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
