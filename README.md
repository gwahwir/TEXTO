# Issue Explorer

An interactive 2D semantic map of policy issues (or any list of short texts). Points are
projected from high-dimensional embeddings (via `sentence-transformers`) into 2D using UMAP,
then rendered in an interactive scatter plot.

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## Your issues file

Edit `issues.txt` — one issue per line, blank lines are ignored. The sample file contains
~170 US policy topics as a starting point.

Example:
```
Universal healthcare coverage
Climate change legislation
Electoral college reform
...
```

First run will:
1. Load the `all-MiniLM-L6-v2` model (~80 MB download on first use)
2. Encode all issues into 384-dimensional vectors
3. Run UMAP to produce 2-D coordinates
4. Cache embeddings to `embeddings_cache.pkl` so subsequent runs are fast

---

## UI Guide

| Interaction | Action |
|-------------|--------|
| **Hover** over a dot | See issue text tooltip + similarity score |
| **Click** a dot | Select / deselect that issue |
| **Click + drag** | Lasso-select a region of dots |
| **Alt + drag** | Pan the map |
| **Scroll wheel** | Zoom in/out (centred on cursor) |
| `+` / `−` / `⌖` buttons | Zoom / reset view |
| **Search box** | Type a category (e.g. `education`) and press Enter |
| **Threshold slider** | 0 → show everything; 0.5 → only strong matches |
| **Clear** button | Reset search and selection |

Sidebar always shows currently selected or matched issues, sorted by similarity score.

---

## Customisation

| What | Where |
|------|-------|
| Swap embedding model | `MODEL_NAME` in `app.py` |
| Number of UMAP neighbours | `n_neighbors` in `get_coords()` |
| Port | `app.run(port=5000)` at the bottom of `app.py` |
| Dot colours / sizes | CSS variables in `<style>` + `C` object in JS |
| Issues file path | `ISSUES_FILE` constant in `app.py` |

---

## Project Structure

```
issue-explorer/
├── app.py                 # Flask backend + embedding/UMAP logic
├── requirements.txt
├── issues.txt             # Your issues (one per line)
├── embeddings_cache.pkl   # Auto-generated on first run
└── templates/
    └── index.html         # All frontend (HTML + CSS + JS, single file)
```
