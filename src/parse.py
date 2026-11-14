"""Code parsing and AST extraction module.

Codebase Archaeologist - Function & Class Parser:
- Parses Python source files into individual functions, methods, and classes using `ast`.
- Extracts name, type, start line, end line, and exact source code for each definition.
- Analyzes git diffs to identify which specific functions and classes were modified.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


def parse_definitions(source_code: str) -> list[dict[str, Any]]:
    """Split Python source code into functions, methods, and classes using ast.

    Args:
        source_code: Full Python source code as a string.

    Returns:
        List of dictionaries with:
        - name (str): Identifier name (e.g., 'send' or 'Session.send').
        - type (str): Definition kind ('function', 'async_function', 'class', or 'method').
        - start_line (int): First line number in source file (1-indexed).
        - end_line (int): Last line number in source file (inclusive, 1-indexed).
        - source_code (str): Exact source code snippet for this definition.
    """
    if not source_code or not source_code.strip():
        return []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        # Gracefully handle files with syntax errors or unsupported constructs
        return []

    definitions: list[dict[str, Any]] = []
    source_lines = source_code.splitlines()

    def get_node_source(node: ast.AST, start: int, end: int) -> str:
        """Extract source code for an AST node with fallback to line slicing."""
        segment = ast.get_source_segment(source_code, node)
        if segment is not None:
            return segment
        # Fallback to slicing lines (1-indexed to 0-indexed)
        return "\n".join(source_lines[max(0, start - 1):end])

    for node in tree.body:
        # 1. Top-level Classes and their interior methods
        if isinstance(node, ast.ClassDef):
            start = node.lineno
            end = node.end_lineno or start
            definitions.append({
                "name": node.name,
                "type": "class",
                "start_line": start,
                "end_line": end,
                "source_code": get_node_source(node, start, end),
            })

            # Inspect methods inside this class
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    item_start = item.lineno
                    item_end = item.end_lineno or item_start
                    is_async = isinstance(item, ast.AsyncFunctionDef)
                    definitions.append({
                        "name": f"{node.name}.{item.name}",
                        "type": "async_method" if is_async else "method",
                        "parent_class": node.name,
                        "start_line": item_start,
                        "end_line": item_end,
                        "source_code": get_node_source(item, item_start, item_end),
                    })

        # 2. Top-level Functions
        elif isinstance(node, ast.FunctionDef):
            start = node.lineno
            end = node.end_lineno or start
            definitions.append({
                "name": node.name,
                "type": "function",
                "start_line": start,
                "end_line": end,
                "source_code": get_node_source(node, start, end),
            })

        # 3. Top-level Async Functions
        elif isinstance(node, ast.AsyncFunctionDef):
            start = node.lineno
            end = node.end_lineno or start
            definitions.append({
                "name": node.name,
                "type": "async_function",
                "start_line": start,
                "end_line": end,
                "source_code": get_node_source(node, start, end),
            })

    return definitions


def parse_file(file_path: str | Path) -> list[dict[str, Any]]:
    """Read a Python file from disk and return its parsed functions and classes."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    content = path.read_text(encoding="utf-8", errors="replace")
    return parse_definitions(content)


def extract_changed_line_numbers(diff_text: str) -> set[int]:
    """Parse a unified git diff and extract all modified line numbers in the new file.

    Analyzes hunk headers (e.g., '@@ -50,5 +52,6 @@') and accounts for:
    - Added or modified lines ('+'): mapped to new file line numbers.
    - Deleted lines ('-'): mapped to the line where deletion occurred in new file.
    - Context lines (' '): advances the line pointer without marking changes.
    """
    changed_lines: set[int] = set()
    current_line = 0

    # Pattern for hunk header: @@ -old_start[,old_len] +new_start[,new_len] @@
    hunk_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")

    for line in diff_text.splitlines():
        hunk_match = hunk_pattern.match(line)
        if hunk_match:
            current_line = int(hunk_match.group(1))
            continue

        if current_line == 0:
            # Preamble before first hunk (diff --git, index, etc.)
            continue

        if line.startswith("+") and not line.startswith("+++"):
            # Line added or modified in the new file version
            changed_lines.add(current_line)
            current_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            # Line deleted from old version; marks the point of deletion in new file
            changed_lines.add(max(1, current_line))
        elif line.startswith(" ") or not line:
            # Unchanged context line
            current_line += 1

    return changed_lines


def find_changed_functions_in_diff(
    source_code: str,
    diff_text: str,
) -> list[dict[str, Any]]:
    """Find all functions, methods, and classes modified by a file's git diff.

    Args:
        source_code: Source code of the file after the commit.
        diff_text: Unified diff string for this file.

    Returns:
        List of changed definitions, each augmented with:
        - changed_lines (list[int]): Sorted list of lines modified within this definition.
    """
    if not source_code or not diff_text:
        return []

    # 1. Parse all functions and classes from the file's AST
    definitions = parse_definitions(source_code)
    if not definitions:
        return []

    # 2. Extract line numbers touched by the diff
    changed_lines = extract_changed_line_numbers(diff_text)
    if not changed_lines:
        return []

    # 3. Match definitions against changed lines
    # Rule: If a changed line falls inside a method, report ONLY the method.
    # Report the class itself only if a modified line is inside the class but
    # outside all of its methods (e.g. class variables, decorators, docstrings, base classes).
    changed_definitions: list[dict[str, Any]] = []

    # Map class name -> set of line numbers covered by its methods
    class_method_lines: dict[str, set[int]] = {}
    for d in definitions:
        if d.get("parent_class"):
            p_cls = d["parent_class"]
            if p_cls not in class_method_lines:
                class_method_lines[p_cls] = set()
            class_method_lines[p_cls].update(range(d["start_line"], d["end_line"] + 1))

    for definition in definitions:
        def_range = range(definition["start_line"], definition["end_line"] + 1)

        if definition["type"] == "class":
            # For classes, exclude lines that belong to interior methods
            interior_method_lines = class_method_lines.get(definition["name"], set())
            overlap = [
                line for line in sorted(changed_lines)
                if line in def_range and line not in interior_method_lines
            ]
        else:
            # Standalone functions and methods
            overlap = [line for line in sorted(changed_lines) if line in def_range]

        if overlap:
            matched = dict(definition)
            matched["changed_lines"] = overlap
            changed_definitions.append(matched)

    return changed_definitions


def find_changed_functions_in_commit(
    commit: Any,
) -> list[dict[str, Any]]:
    """Identify all modified functions across all Python files in a PyDriller Commit.

    Args:
        commit: A PyDriller Commit object.

    Returns:
        List of dictionaries with:
        - file (str): Relative file path.
        - name (str): Function, method, or class name.
        - type (str): 'class', 'method', 'function', or 'async_function'.
        - start_line (int): Start line number in new file.
        - end_line (int): End line number in new file.
        - changed_lines (list[int]): Changed line numbers within this definition.
        - source_code (str): Code segment of the changed definition.
    """
    all_changed: list[dict[str, Any]] = []

    for mod in commit.modified_files:
        file_path = mod.new_path or mod.old_path or mod.filename
        if not file_path or not file_path.endswith(".py"):
            continue

        # Prefer post-commit source code; fallback to pre-commit for deleted files
        source = mod.source_code or mod.source_code_before
        diff = mod.diff or ""

        if not source or not diff:
            continue

        changed_defs = find_changed_functions_in_diff(source, diff)
        for item in changed_defs:
            record = dict(item)
            record["file"] = file_path
            all_changed.append(record)

    return all_changed


if __name__ == "__main__":
    from pydriller import Repository

    project_root = Path(__file__).resolve().parent.parent
    sample_repo_path = project_root / "repos" / "requests"

    print("=" * 75)
    print("Codebase Archaeologist - AST Function & Class Parser Demo")
    print("=" * 75)

    if not sample_repo_path.exists() or not (sample_repo_path / ".git").exists():
        print(f"Sample repo not found at: {sample_repo_path.resolve()}")
        print("Please ensure repos/requests is cloned before running the demo.")
        exit(1)

    print(f"Analyzing commits from: {sample_repo_path.resolve()}\n")

    # Traverse commits and test on 5 commits that modified Python files
    repo = Repository(str(sample_repo_path), order="reverse")
    commits_tested = 0

    for commit in repo.traverse_commits():
        # Check if commit touched any Python files with diffs and source code
        py_mods = [
            m for m in commit.modified_files
            if (m.filename or "").endswith(".py") and m.diff and (m.source_code or m.source_code_before)
        ]
        if not py_mods:
            continue

        commits_tested += 1
        short_hash = commit.hash[:8]
        first_line = commit.msg.splitlines()[0] if commit.msg else "(no message)"
        date_str = commit.author_date.strftime("%Y-%m-%d") if commit.author_date else ""

        print(f"[{commits_tested}/5] Commit {short_hash} ({date_str}) by {commit.author.name}")
        print(f"      Message: {first_line}")

        # Find all changed functions in this commit
        changed_functions = find_changed_functions_in_commit(commit)

        if not changed_functions:
            print("      No function-level changes detected (e.g. module imports or global variables).")
        else:
            print(f"      Changed definitions found ({len(changed_functions)}):")
            for item in changed_functions:
                lines_preview = f"lines {item['start_line']}-{item['end_line']}"
                touched_preview = f"modified {len(item['changed_lines'])} line(s): {item['changed_lines'][:4]}"
                if len(item['changed_lines']) > 4:
                    touched_preview += "..."
                print(f"        * [{item['type'].upper()}] {item['file']} -> {item['name']} ({lines_preview}, {touched_preview})")

                # Show a 2-line snippet of the changed function
                first_lines = item["source_code"].splitlines()[:2]
                snippet = "          " + "\n          ".join(first_lines)
                print(f"{snippet}")

        print("-" * 75)

        if commits_tested >= 5:
            break

    print("AST parsing and diff mapping test completed successfully!")
    print("=" * 75)
