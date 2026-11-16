"""One-shot patch: replace the old ask() body with a thin wrapper."""
import pathlib, re, sys

src = pathlib.Path("src/qa.py")
text = src.read_text(encoding="utf-8")

# Locate the old ask() signature line
sig = "def ask(\n    question: str,\n    chroma_dir: str | Path = CHROMA_DIR,\n    top_k: int = TOP_K_COMBINED,\n    force_structured: bool = False,\n    verbose: bool = True,\n) -> str:"

if sig not in text:
    print("ERROR: could not find ask() signature — aborting")
    sys.exit(1)

# Find where the function body starts (after the signature)
sig_end = text.index(sig) + len(sig)

# Find the NEXT top-level function/class definition after ask()
# so we know where ask() ends
# (the next top-level def starts at column 0)
after_sig = text[sig_end:]
# match next top-level def or class
m = re.search(r"\n\n\n# -+\n", after_sig)
if not m:
    m = re.search(r"\ndef [a-z]", after_sig)
if not m:
    print("ERROR: could not find end of ask() — aborting")
    sys.exit(1)

end_of_ask = sig_end + m.start()

old_body = text[sig_end:end_of_ask]

new_body = '''
    """Thin wrapper around ask_detailed() that returns only the answer string.

    Plain English:
    Calls ask_detailed() to do all the work (retrieval, abstain check, Gemini,
    fallback, citation validation, timing), prints the same log lines to stdout
    when verbose=True, and returns just the answer text.
    All callers keep working without any changes.
    """
    result = ask_detailed(
        question,
        chroma_dir=chroma_dir,
        top_k=top_k,
        force_structured=force_structured,
        verbose=verbose,
    )
    return result["answer"]
'''

new_text = text[:sig_end] + new_body + text[end_of_ask:]
src.write_text(new_text, encoding="utf-8")
print(f"OK — replaced {len(old_body)} chars with {len(new_body)} chars")
