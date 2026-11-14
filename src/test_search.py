"""Quick search test for Codebase Archaeologist.

Runs three natural-language queries against the ChromaDB index and prints
the top 5 results for each.  Run AFTER index.py has built the index:

    python -X utf8 src/test_search.py

Plain English: We import the search() function from index.py and pass it
three questions that a developer might actually ask when debugging.  The
output tells us whether our embedding + PR enrichment is good enough to
surface the right commits even when the query doesn't use the exact words
from the commit message.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from either the project root or the src/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from index import search, _print_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR   = PROJECT_ROOT / "chroma_db"

QUERIES: list[str] = [
    "why did uploading a temp file break",
    "why were hasattr checks added",
    "what problem did file-like proxy objects cause",
]

if __name__ == "__main__":
    print("=" * 72)
    print("Codebase Archaeologist - Search Test")
    print(f"ChromaDB: {CHROMA_DIR}")
    print("=" * 72)

    for query in QUERIES:
        results = search(query, n=5, chroma_dir=CHROMA_DIR)
        _print_results(query, results)

    print("\nDone.")
