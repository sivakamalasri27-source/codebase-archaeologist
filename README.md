# Codebase Archaeologist 🔍📜

Codebase Archaeologist is an intelligent developer tool that answers *"Why was this code changed?"* questions about a GitHub repository. It investigates git history (commits, diffs, PR descriptions, and issue discussions) alongside the code itself, providing rigorous natural language explanations backed by concrete commit hashes and PR citations.

---

## 🛠️ Pipeline Roadmap

1. **Week 1 (Completed)**:
   - Extract git commits, metadata, and Python file diffs using PyDriller.
   - Fetch PR descriptions and issue discussions via GitHub REST API with local caching.
2. **Week 2 (Completed)**:
   - Parse Python code with Python's `ast` module into function & class definitions.
   - Map unified git diffs to specific modified functions, methods, and classes.
3. **Week 3**:
   - Vector indexing and storage with ChromaDB.
4. **Week 4**:
   - LLM-powered Q&A retrieval with verified commit citations.
5. **Week 5**:
   - Streamlit interactive dashboard with Plotly timelines.

---

## 📂 Project Structure

```text
codebase-archaeologist/
├── app.py                  # Streamlit web UI (Week 5)
├── benchmark/              # Evaluation benchmarks and test datasets
│   └── README.md
├── requirements.txt        # Project dependencies
├── .env.example            # Template for GitHub credentials
├── README.md               # Project documentation
└── src/
    ├── __init__.py
    ├── extract.py          # Git commit extraction & GitHub API client (Week 1)
    ├── parse.py            # AST parsing and function tracking (Week 2)
    ├── index.py            # ChromaDB embeddings & vector storage (Week 3)
    ├── qa.py               # Citation-backed Q&A pipeline (Week 4)
    └── classify.py         # Commit categorization & intent classification
```

---

## 🚀 Quickstart & Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure GitHub Token (Optional but Recommended)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your GitHub Personal Access Token to avoid hitting GitHub's unauthenticated 60 requests/hour rate limit.

### 3. Run Week 1 Extraction Demo
```bash
python src/extract.py
```
This will inspect a local clone of `psf/requests` and print out the first 10 commits with changed Python files and diff snippets.
