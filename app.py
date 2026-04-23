"""
Textonomy – Issue Explorer + Reports
=====================================
Routes:
  /explorer  – scatter plot explorer
  /reports   – report analysis page
  /          – redirects to /explorer

Category filter schema (categories.json):
  Each category has a `filters` list. Each filter:
    { "query": str, "threshold": float, "mode": "or"|"and"|"not", "exact": bool }
  `exact: true` means plain substring match instead of semantic similarity.
  Legacy format (single query/threshold at top level) is auto-upgraded on load.
  A point is IN the category when the combined filter logic is satisfied:
    - Start with the set of OR-mode filters united together
    - Intersect with every AND-mode filter
    - Subtract every NOT-mode filter
  If there are no OR/AND filters, nothing matches (empty set).
"""

import json, pickle, re, unicodedata, time, threading
from pathlib import Path
import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for
import training

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
ISSUES_FILE     = "issues.txt"
CATEGORIES_FILE = "categories.json"
REPORTS_DIR     = "reports"
DEFAULT_MAX_DISPLAY = 1000
FEEDBACK_FILE   = "user_feedback.jsonl"
DEFAULT_TRAINING_THRESHOLD = 50

# ── Globals ───────────────────────────────────────────────────────────────────
_models      = {}
_caches      = {}
_coords      = {}
_display_ids = None
_issues      = None
_active_model = "all-MiniLM-L6-v2"
_available_models = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2",
]
_training_jobs = {}
_model_paths   = {}  # name -> local path for fine-tuned models


def load_issues():
    path = Path(ISSUES_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {ISSUES_FILE}")
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def get_model(model_name=None):
    global _active_model
    if model_name is None:
        model_name = _active_model
    if model_name not in _models:
        from sentence_transformers import SentenceTransformer
        # Use local path if registered, otherwise treat as HuggingFace model ID
        load_path = _model_paths.get(model_name, model_name)
        print(f"[info] Loading model '{model_name}' from '{load_path}' ...")
        _models[model_name] = SentenceTransformer(load_path)
        print("[info] Model ready.")
    return _models[model_name]


def cache_file_for(model_name):
    safe = model_name.replace("/", "_").replace("-", "_")
    return f"embeddings_cache_{safe}.pkl"


def get_embeddings(issues, model_name=None):
    if model_name is None:
        model_name = _active_model
    cache_path = Path(cache_file_for(model_name))
    cached = _caches.get(model_name)
    if cached is not None and cached.get("issues") == issues:
        return cached["embeddings"]

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            saved = pickle.load(f)
        saved_issues = saved.get("issues", [])
        saved_embs = np.array(saved.get("embeddings", []), dtype=np.float32)
        if saved_issues == issues:
            print(f"[info] Embeddings loaded from cache ({model_name}).")
            _caches[model_name] = {"issues": issues, "embeddings": saved_embs}
            return saved_embs
        saved_map = {iss: emb for iss, emb in zip(saved_issues, saved_embs)}
        new_issues = [iss for iss in issues if iss not in saved_map]
        if new_issues:
            print(f"[info] Encoding {len(new_issues)} new issues (cached {len(saved_map)}) ...")
            new_embs = get_model(model_name).encode(
                new_issues, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
            for iss, emb in zip(new_issues, new_embs):
                saved_map[iss] = emb
        embs = np.array([saved_map[iss] for iss in issues], dtype=np.float32)
        with open(cache_path, "wb") as f:
            pickle.dump({"issues": issues, "embeddings": embs.tolist()}, f)
        _caches[model_name] = {"issues": issues, "embeddings": embs}
        return embs

    print(f"[info] Encoding {len(issues)} issues ({model_name}) ...")
    embs = get_model(model_name).encode(
        issues, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    with open(cache_path, "wb") as f:
        pickle.dump({"issues": issues, "embeddings": embs.tolist()}, f)
    _caches[model_name] = {"issues": issues, "embeddings": embs}
    return embs


def get_coords(embeddings, model_name=None):
    if model_name is None:
        model_name = _active_model
    if model_name in _coords and _coords[model_name] is not None:
        cached_embs = _caches.get(model_name, {}).get("embeddings")
        if cached_embs is not None and cached_embs.shape == embeddings.shape:
            return _coords[model_name]
    print("[info] Running PCA pre-processing ...")
    from sklearn.decomposition import PCA
    n_pca = min(50, embeddings.shape[0], embeddings.shape[1])
    pca_embs = PCA(n_components=n_pca, random_state=42).fit_transform(embeddings)
    print("[info] Running UMAP ...")
    import umap
    reducer = umap.UMAP(
        n_components=2, random_state=42, min_dist=0.05,
        n_neighbors=min(30, len(embeddings) - 1), metric="cosine")
    coords = reducer.fit_transform(pca_embs).astype(np.float32)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    coords = (coords - lo) / (hi - lo + 1e-9)
    _coords[model_name] = coords
    return coords


def cosine_scores(q_emb, embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    q_norm = float(np.linalg.norm(q_emb))
    return (embeddings @ q_emb) / (norms * q_norm + 1e-9)


def sample_display_ids(n_total, max_display, seed=42):
    if n_total <= max_display:
        return list(range(n_total))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(n_total, size=max_display, replace=False).tolist())


def get_active_embeddings():
    return _caches.get(_active_model, {}).get("embeddings")


def get_active_coords():
    return _coords.get(_active_model)


def initialise():
    global _issues, _display_ids, _active_model
    _active_model = get_user_setting("active_model", "all-MiniLM-L6-v2")
    # Register local path for fine-tuned models so get_model() can find them
    local_path = Path("trained_models") / _active_model
    if local_path.exists():
        _model_paths[_active_model] = str(local_path)
    _issues = load_issues()
    embs = get_embeddings(_issues)
    get_coords(embs)
    max_d = get_user_setting("max_display", DEFAULT_MAX_DISPLAY)
    seed = get_user_setting("display_seed", 42)
    _display_ids = sample_display_ids(len(_issues), max_d, seed)
    print(f"[info] Ready – {len(_issues)} issues, {len(_display_ids)} displayed.")


# ── User settings ─────────────────────────────────────────────────────────────
SETTINGS_FILE = "settings.json"

def get_user_setting(key, default):
    try:
        s = json.loads(Path(SETTINGS_FILE).read_text(encoding="utf-8"))
        return s.get(key, default)
    except Exception:
        return default

def save_user_setting(key, value):
    try:
        path = Path(SETTINGS_FILE)
        s = {}
        if path.exists():
            try: s = json.loads(path.read_text(encoding="utf-8"))
            except Exception: pass
        s[key] = value
        path.write_text(json.dumps(s, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[warn] save_user_setting: {e}")


def get_training_threshold():
    """Get the feedback count threshold for auto-training."""
    return get_user_setting("training_threshold", DEFAULT_TRAINING_THRESHOLD)


# ── Multi-filter category logic ───────────────────────────────────────────────

def normalize_filters(cat_dict):
    """
    Upgrade legacy single-filter format to multi-filter format.
    Returns list of filter dicts with keys: query, threshold, mode, exact
    """
    if "filters" in cat_dict and isinstance(cat_dict["filters"], list):
        out = []
        for f in cat_dict["filters"]:
            out.append({
                "query": f.get("query", ""),
                "threshold": float(f.get("threshold", 0.3)),
                "mode": f.get("mode", "or"),
                "exact": bool(f.get("exact", False)),
            })
        return out
    # Legacy: single query/threshold at top level
    return [{
        "query": cat_dict.get("query", ""),
        "threshold": float(cat_dict.get("threshold", 0.3)),
        "mode": "or",
        "exact": False,
    }]


def compute_filter_match_set(filt, all_issues, embeddings, display_ids):
    """
    Returns set of display-local ids (0..len(display_ids)-1) that match this filter.
    """
    query = filt["query"].strip()
    threshold = filt["threshold"]
    exact = filt.get("exact", False)

    if exact:
        # Plain substring match on issue text
        q_lower = query.lower()
        matched = set()
        for i, orig_i in enumerate(display_ids):
            if q_lower in all_issues[orig_i].lower():
                matched.add(i)
        return matched
    else:
        if not query:
            return set()
        q_emb = get_model().encode([query], convert_to_numpy=True)[0].astype(np.float32)
        sims = cosine_scores(q_emb, embeddings)
        return {i for i, orig_i in enumerate(display_ids) if sims[orig_i] >= threshold}


def compute_filter_match_set_full(filt, all_issues, embeddings):
    """
    Returns set of global issue indices (0..len(all_issues)-1) for full-corpus matching.
    Used by reports analysis.
    """
    query = filt["query"].strip()
    threshold = filt["threshold"]
    exact = filt.get("exact", False)

    if exact:
        q_lower = query.lower()
        return {i for i, iss in enumerate(all_issues) if q_lower in iss.lower()}
    else:
        if not query:
            return set()
        q_emb = get_model().encode([query], convert_to_numpy=True)[0].astype(np.float32)
        sims = cosine_scores(q_emb, embeddings)
        return {i for i in range(len(all_issues)) if sims[i] >= threshold}


def apply_filter_logic(filters, all_issues, embeddings, display_ids):
    """
    Apply multi-filter boolean logic and return dict of {display_id_str: best_score}.
    Logic: union of OR filters, intersected by AND filters, minus NOT filters.
    Score is best OR/AND similarity; exact matches get score 1.0.
    """
    or_sets = []
    and_sets = []
    not_sets = []

    for filt in filters:
        mode = filt.get("mode", "or")
        matched = compute_filter_match_set(filt, all_issues, embeddings, display_ids)
        if mode == "not":
            not_sets.append(matched)
        elif mode == "and":
            and_sets.append(matched)
        else:
            or_sets.append(matched)

    if not or_sets and not and_sets:
        return {}

    # Start with union of OR sets (or all if only AND sets)
    if or_sets:
        result = set().union(*or_sets)
    else:
        result = set(range(len(display_ids)))

    for s in and_sets:
        result &= s
    for s in not_sets:
        result -= s

    # Build score dict: best semantic score across OR/AND filters for each matched id
    scores = {}
    for i in result:
        orig_i = display_ids[i]
        best = 0.0
        for filt in filters:
            if filt.get("mode", "or") == "not":
                continue
            if filt.get("exact", False):
                q_lower = filt["query"].strip().lower()
                if q_lower in all_issues[orig_i].lower():
                    best = max(best, 1.0)
            else:
                q = filt["query"].strip()
                if q:
                    q_emb = get_model().encode([q], convert_to_numpy=True)[0].astype(np.float32)
                    sims = cosine_scores(q_emb, embeddings)
                    best = max(best, float(sims[orig_i]))
        scores[str(i)] = best

    return scores


def apply_filter_logic_full(filters, all_issues, embeddings):
    """
    Same as above but returns set of global issue indices for reports.
    """
    or_sets = []
    and_sets = []
    not_sets = []

    for filt in filters:
        mode = filt.get("mode", "or")
        matched = compute_filter_match_set_full(filt, all_issues, embeddings)
        if mode == "not":
            not_sets.append(matched)
        elif mode == "and":
            and_sets.append(matched)
        else:
            or_sets.append(matched)

    if not or_sets and not and_sets:
        return set()

    if or_sets:
        result = set().union(*or_sets)
    else:
        result = set(range(len(all_issues)))

    for s in and_sets:
        result &= s
    for s in not_sets:
        result -= s

    return result


def rescore_category(cat_dict):
    """Build full category response dict including scores for display points."""
    embs = get_active_embeddings()
    filters = normalize_filters(cat_dict)
    scores = apply_filter_logic(filters, _issues, embs, _display_ids)
    return {
        "label": cat_dict.get("label", ""),
        "color": cat_dict.get("color", "#3a6fff"),
        "active": cat_dict.get("active", True),
        "filters": filters,
        # Legacy fields for backward compat
        "query": filters[0]["query"] if filters else "",
        "threshold": filters[0]["threshold"] if filters else 0.3,
        "scores": scores,
    }


def load_persisted_categories():
    path = Path(CATEGORIES_FILE)
    if not path.exists():
        return []
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] Could not load categories: {e}")
        return []
    result = []
    for cat in saved:
        try:
            c = rescore_category(cat)
            result.append(c)
            print(f"[info] Loaded category '{c['label']}': {len(c['scores'])} matches.")
        except Exception as e:
            print(f"[warn] Skipping category '{cat.get('label', '?')}': {e}")
    return result


def save_categories_to_disk(categories):
    to_save = []
    for c in categories:
        to_save.append({
            "label": c["label"],
            "color": c["color"],
            "active": c["active"],
            "filters": c.get("filters", [{"query": c.get("query",""), "threshold": c.get("threshold",0.3), "mode":"or","exact":False}]),
        })
    Path(CATEGORIES_FILE).write_text(json.dumps(to_save, indent=2), encoding="utf-8")


# ── Reports helpers ───────────────────────────────────────────────────────────

def load_reports():
    rdir = Path(REPORTS_DIR)
    if not rdir.exists():
        return []
    reports = []
    for p in sorted(rdir.glob("*.txt")):
        body = p.read_text(encoding="utf-8")
        lines = body.strip().splitlines()
        title = lines[0].replace("Title:", "").strip() if lines and lines[0].startswith("Title:") else p.stem
        if lines and lines[0].startswith("Title:"):
            body = "\n".join(lines[1:]).lstrip("\n")
        reports.append({"id": p.stem, "filename": p.name, "title": title, "body": body})
    return reports


def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def match_issues_in_text(text, issues):
    norm_text = normalize_text(text)
    matched = []
    for issue in issues:
        norm_issue = normalize_text(issue)
        if norm_issue in norm_text:
            matched.append(issue)
    return matched


def build_report_analysis():
    path = Path(CATEGORIES_FILE)
    raw_cats = []
    if path.exists():
        try:
            raw_cats = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    embs = get_active_embeddings()
    cat_issue_sets = []
    for cat in raw_cats:
        filters = normalize_filters(cat)
        matched_global = apply_filter_logic_full(filters, _issues, embs)
        matched_issue_texts = {_issues[i] for i in matched_global}
        cat_issue_sets.append({
            "label": cat.get("label", cat.get("query", "")),
            "color": cat.get("color", "#3a6fff"),
            "active": cat.get("active", True),
            "issue_texts": matched_issue_texts,
        })

    reports = load_reports()
    result = []
    for r in reports:
        matched_issues = match_issues_in_text(r["body"], _issues)
        matched_issue_set = set(matched_issues)
        matched_categories = []
        for cat in cat_issue_sets:
            overlap = matched_issue_set & cat["issue_texts"]
            if overlap:
                matched_categories.append({
                    "label": cat["label"],
                    "color": cat["color"],
                    "active": cat["active"],
                    "matched_issues": sorted(overlap),
                })
        result.append({
            "id": r["id"],
            "title": r["title"],
            "body": r["body"],
            "matched_issues": matched_issues,
            "matched_categories": matched_categories,
        })
    return result


def highlight_text(body, issues, categories):
    issue_color = {}
    issue_categories = {}
    for cat in categories:
        for issue in cat["matched_issues"]:
            if issue not in issue_color:
                issue_color[issue] = cat["color"]
                issue_categories[issue] = []
            issue_categories[issue].append(cat["label"])

    sorted_issues = sorted(issue_color.keys(), key=len, reverse=True)
    html = body
    placeholders = {}
    for idx, issue in enumerate(sorted_issues):
        color = issue_color[issue]
        cats_str = ", ".join(issue_categories[issue])
        placeholder = f"__ISSUE_{idx}__"
        span = (f'<mark class="issue-highlight" style="background:{color}33;" '
                f'data-issue="{issue}" data-cats="{cats_str}" data-color="{color}">{issue}</mark>')
        placeholders[placeholder] = span
        norm_issue = normalize_text(issue)
        escaped_words = [re.escape(w) for w in norm_issue.split()]
        pattern_str = r"\s+".join(escaped_words)
        pattern = re.compile(pattern_str, re.IGNORECASE)
        html = pattern.sub(placeholder, html)

    for ph, span in placeholders.items():
        html = html.replace(ph, span)

    html = html.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f"<p>{html}</p>"


# ── Continual Learning Routes ────────────────────────────────────────────────

MAX_TRAINED_MODELS = 3

def _prepare_model_output_dir(model_name):
    """Ensure output dir exists and enforce the 3-model limit (drop oldest)."""
    import re
    safe_name = re.sub(r'[^\w\-]', '_', model_name)
    models_dir = Path(training.TRAINED_MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir = models_dir / safe_name
    # Enforce limit: remove oldest models if already at max
    existing = training.list_trained_models()
    existing = [m for m in existing if m["name"] != safe_name]  # exclude current target
    while len(existing) >= MAX_TRAINED_MODELS:
        oldest = existing.pop()  # list is sorted newest-first
        import shutil
        shutil.rmtree(oldest["path"], ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@app.route("/api/feedback/ranking", methods=["POST"])
def api_feedback_ranking():
    """Record user's ranking of search results. Auto-trigger training at threshold."""
    data = request.get_json()
    query = data.get("query", "")
    ranked_issues = data.get("ranked_issues", [])
    threshold = data.get("threshold", 0.5)

    feedback = {
        "timestamp": time.time(),
        "query": query,
        "threshold": threshold,
        "ranked_issues": ranked_issues,
        "model_version": _active_model,
    }

    with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(feedback) + '\n')

    # Count total feedback
    feedback_count = sum(1 for _ in open(FEEDBACK_FILE, 'r', encoding='utf-8'))

    # Auto-trigger training at configurable threshold
    training_threshold = get_training_threshold()
    should_auto_train = feedback_count % training_threshold == 0 and feedback_count >= training_threshold
    training_triggered = False

    if should_auto_train:
        # Check if no training currently running
        has_running = any(j.get("status") == "running" for j in _training_jobs.values())
        if not has_running:
            # Trigger training automatically
            def auto_train():
                try:
                    model_name = f"{_active_model}-fine-tuned"
                    output_dir = _prepare_model_output_dir(model_name)
                    job_id = "finetuned_model"
                    _training_jobs[job_id] = {"status": "running", "progress": 0.0}
                    base_model = _model_paths.get(_active_model, _active_model)
                    training.train_model(base_model, str(output_dir), learning_rate=2e-5, epochs=1)
                    _training_jobs[job_id] = {"status": "completed", "progress": 1.0, "model_path": str(output_dir)}
                except Exception as e:
                    _training_jobs[job_id] = {"status": "failed", "error": str(e)}

            thread = threading.Thread(target=auto_train, daemon=True)
            thread.start()
            training_triggered = True

    return jsonify({
        "ok": True,
        "feedback_count": feedback_count,
        "training_triggered": training_triggered,
        "training_threshold": training_threshold,
        "next_training_at": ((feedback_count // training_threshold) + 1) * training_threshold
    })


@app.route("/api/feedback/stats", methods=["GET"])
def api_feedback_stats():
    """Get feedback statistics."""
    path = Path(FEEDBACK_FILE)
    if not path.exists():
        return jsonify({"total_queries": 0, "total_rankings": 0})

    feedback = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    feedback.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not feedback:
        return jsonify({"total_queries": 0, "total_rankings": 0})

    total_rankings = sum(len(entry.get("ranked_issues", [])) for entry in feedback)

    return jsonify({
        "total_queries": len(feedback),
        "total_rankings": total_rankings,
        "oldest": min(entry.get("timestamp", 0) for entry in feedback),
        "newest": max(entry.get("timestamp", 0) for entry in feedback),
    })


@app.route("/api/training/start", methods=["POST"])
def api_training_start():
    """Start a fine-tuning job (manual trigger or auto-triggered)."""
    data = request.get_json()
    min_feedback = data.get("min_feedback_count", 10)
    learning_rate = data.get("learning_rate", 2e-5)
    epochs = data.get("epochs", 1)
    model_name = data.get("model_name", "").strip() or f"{_active_model}-fine-tuned"

    # Check feedback count
    stats = training.load_feedback_data()
    if len(stats) < min_feedback:
        return jsonify({
            "error": f"Not enough feedback: {len(stats)} < {min_feedback}"
        }), 400

    # Check if training already running
    for job_id, status in _training_jobs.items():
        if status.get("status") == "running":
            return jsonify({"error": "Training already in progress"}), 409

    job_id = "finetuned_model"

    # Start training in background
    def train_worker():
        try:
            output_dir = _prepare_model_output_dir(model_name)
            _training_jobs[job_id] = {"status": "running", "progress": 0.0}
            base_model = _model_paths.get(_active_model, _active_model)
            training.train_model(
                base_model,
                str(output_dir),
                learning_rate=learning_rate,
                epochs=epochs
            )
            _training_jobs[job_id] = {"status": "completed", "progress": 1.0, "model_path": str(output_dir)}
        except Exception as e:
            _training_jobs[job_id] = {"status": "failed", "error": str(e)}
            print(f"[error] Training failed: {e}")

    thread = threading.Thread(target=train_worker, daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "status": "started"})


@app.route("/api/training/status/<job_id>", methods=["GET"])
def api_training_status(job_id):
    """Check training job status."""
    if job_id not in _training_jobs:
        return jsonify({"error": "Job not found"}), 404

    return jsonify(_training_jobs[job_id])


@app.route("/api/models/trained", methods=["GET"])
def api_list_trained_models():
    """List all fine-tuned models."""
    models = training.list_trained_models()
    return jsonify({"models": models})


@app.route("/api/models/load", methods=["POST"])
def api_load_trained_model():
    """Load a specific trained model."""
    global _active_model, _models, _caches, _coords

    data = request.get_json()
    model_path = data.get("model_path", "")

    if not Path(model_path).exists():
        return jsonify({"error": "Model not found"}), 404

    try:
        from sentence_transformers import SentenceTransformer

        # Load the fine-tuned model
        model_name = Path(model_path).name
        _model_paths[model_name] = model_path
        print(f"[info] Loading fine-tuned model from {model_path}")
        _models[model_name] = SentenceTransformer(model_path)
        _active_model = model_name
        save_user_setting("active_model", model_name)

        # Re-compute embeddings and coordinates
        print(f"[info] Re-computing embeddings with fine-tuned model...")
        _caches.clear()
        _coords.clear()
        embs = get_embeddings(_issues, model_name)
        get_coords(embs, model_name)

        return jsonify({"ok": True, "active_model": model_name})
    except Exception as e:
        print(f"[error] Failed to load model: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/delete", methods=["POST"])
def api_delete_trained_model():
    """Delete a specific trained model."""
    import shutil

    data = request.get_json()
    model_path = data.get("model_path", "")

    if not model_path:
        return jsonify({"error": "No model path provided"}), 400

    model_path = Path(model_path)

    if not model_path.exists():
        return jsonify({"error": "Model not found"}), 404

    # Don't allow deleting if it's currently active
    model_name = model_path.name
    if model_name == _active_model:
        return jsonify({"error": "Cannot delete active model. Switch to base model first."}), 400

    try:
        # Remove the model directory
        shutil.rmtree(model_path)
        print(f"[info] Deleted model: {model_path}")

        # Clean up from cache if loaded
        if model_name in _models:
            del _models[model_name]
        if model_name in _model_paths:
            del _model_paths[model_name]

        return jsonify({"ok": True, "deleted": model_name})
    except Exception as e:
        print(f"[error] Failed to delete model: {e}")
        return jsonify({"error": str(e)}), 500


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def root():
    return redirect(url_for("explorer"))

@app.route("/explorer")
def explorer():
    return render_template("explorer.html")

@app.route("/reports")
def reports_page():
    return render_template("reports.html")

@app.route("/api/points")
def api_points():
    coords = get_active_coords()
    points = [
        {"id": i, "x": float(coords[orig_i, 0]), "y": float(coords[orig_i, 1]), "text": _issues[orig_i]}
        for i, orig_i in enumerate(_display_ids)
    ]
    return jsonify(points)

@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    query = data.get("query", "").strip()
    threshold = float(data.get("threshold", 0.5))
    exact = bool(data.get("exact", False))
    if not query:
        return jsonify({"scores": []})
    embs = get_active_embeddings()
    if exact:
        q_lower = query.lower()
        results = [
            {"id": i, "score": 1.0}
            for i, orig_i in enumerate(_display_ids)
            if q_lower in _issues[orig_i].lower()
        ]
    else:
        q_emb = get_model().encode([query], convert_to_numpy=True)[0].astype(np.float32)
        sims = cosine_scores(q_emb, embs)
        results = [
            {"id": i, "score": float(sims[orig_i])}
            for i, orig_i in enumerate(_display_ids)
            if sims[orig_i] >= threshold
        ]
    return jsonify({"scores": results})

@app.route("/api/categories", methods=["GET"])
def api_get_categories():
    return jsonify(load_persisted_categories())

@app.route("/api/categories/add", methods=["POST"])
def api_add_category():
    data = request.get_json()
    # Accept either multi-filter format or legacy single query
    filters = data.get("filters")
    if not filters:
        query = data.get("query", "").strip()
        threshold = float(data.get("threshold", 0.3))
        exact = bool(data.get("exact", False))
        filters = [{"query": query, "threshold": threshold, "mode": "or", "exact": exact}]
    color = data.get("color", "#3a6fff")
    label = data.get("label", filters[0]["query"] if filters else "")
    active = data.get("active", True)

    cat_dict = {"label": label, "color": color, "active": active, "filters": filters}
    cat = rescore_category(cat_dict)
    path = Path(CATEGORIES_FILE)
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing.append({"label": label, "color": color, "active": active, "filters": filters})
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return jsonify(cat)

@app.route("/api/categories/update", methods=["POST"])
def api_update_category():
    """Full update of a category (label, color, active, filters). Identified by old_label."""
    data = request.get_json()
    old_label = data.get("old_label", "")
    new_cat = data.get("category", {})
    path = Path(CATEGORIES_FILE)
    if path.exists():
        try:
            cats = json.loads(path.read_text(encoding="utf-8"))
            for i, c in enumerate(cats):
                if c.get("label") == old_label:
                    cats[i] = {
                        "label": new_cat.get("label", c["label"]),
                        "color": new_cat.get("color", c.get("color", "#3a6fff")),
                        "active": new_cat.get("active", c.get("active", True)),
                        "filters": new_cat.get("filters", normalize_filters(c)),
                    }
                    break
            path.write_text(json.dumps(cats, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[warn] update category: {e}")
            return jsonify({"error": str(e)}), 500
    # Return rescored category
    rescored = rescore_category({
        "label": new_cat.get("label", old_label),
        "color": new_cat.get("color", "#3a6fff"),
        "active": new_cat.get("active", True),
        "filters": new_cat.get("filters", []),
    })
    return jsonify(rescored)

@app.route("/api/categories/rename", methods=["POST"])
def api_rename_category():
    data = request.get_json()
    old_label = data.get("old_label", "")
    new_label = data.get("new_label", "").strip()
    if not new_label:
        return jsonify({"error": "empty label"}), 400
    path = Path(CATEGORIES_FILE)
    if path.exists():
        try:
            cats = json.loads(path.read_text(encoding="utf-8"))
            for c in cats:
                if c.get("label") == old_label:
                    c["label"] = new_label
            path.write_text(json.dumps(cats, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[warn] rename category: {e}")
    return jsonify({"ok": True})

@app.route("/api/categories/delete", methods=["POST"])
def api_delete_category():
    data = request.get_json()
    label = data.get("label", "")
    path = Path(CATEGORIES_FILE)
    if path.exists():
        try:
            cats = json.loads(path.read_text(encoding="utf-8"))
            cats = [c for c in cats if c.get("label") != label]
            path.write_text(json.dumps(cats, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[warn] delete category: {e}")
    return jsonify({"ok": True})

@app.route("/api/categories/import", methods=["POST"])
def api_import_categories():
    data = request.get_json()
    new_cats = data.get("categories", [])
    mode = data.get("mode", "replace")
    prefix = data.get("prefix", "")
    path = Path(CATEGORIES_FILE)
    existing = []
    if path.exists() and mode == "add":
        try: existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception: pass
    for cat in new_cats:
        if prefix:
            cat["label"] = prefix + cat.get("label", "")
        # Upgrade legacy format
        if "filters" not in cat:
            cat["filters"] = normalize_filters(cat)
    merged = existing + new_cats if mode == "add" else new_cats
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return jsonify({"ok": True, "count": len(merged)})

@app.route("/api/categories/export", methods=["GET"])
def api_export_categories():
    path = Path(CATEGORIES_FILE)
    if not path.exists():
        return jsonify([])
    try:
        cats = json.loads(path.read_text(encoding="utf-8"))
        return jsonify(cats)
    except Exception:
        return jsonify([])

@app.route("/api/settings", methods=["GET"])
def api_get_settings():
    return jsonify({
        "max_display": get_user_setting("max_display", DEFAULT_MAX_DISPLAY),
        "active_model": get_user_setting("active_model", "all-MiniLM-L6-v2"),
        "available_models": _available_models,
        "total_issues": len(_issues) if _issues else 0,
        "display_seed": get_user_setting("display_seed", 42),
        "training_threshold": get_training_threshold(),
    })

@app.route("/api/settings", methods=["POST"])
def api_save_settings():
    global _display_ids, _active_model
    data = request.get_json()
    changed = False
    if "max_display" in data:
        val = max(100, min(int(data["max_display"]), len(_issues)))
        save_user_setting("max_display", val)
        changed = True
    if "display_seed" in data:
        save_user_setting("display_seed", int(data["display_seed"]))
        changed = True
    if "training_threshold" in data:
        val = max(10, min(500, int(data["training_threshold"])))  # Clamp 10-500
        save_user_setting("training_threshold", val)
        # No need to set changed = True for this setting
    if "active_model" in data:
        model_name = data["active_model"]
        if model_name in _available_models:
            _active_model = model_name
            save_user_setting("active_model", model_name)
            embs = get_embeddings(_issues, model_name)
            get_coords(embs, model_name)
            changed = True
    if changed:
        max_d = get_user_setting("max_display", DEFAULT_MAX_DISPLAY)
        seed = get_user_setting("display_seed", 42)
        _display_ids = sample_display_ids(len(_issues), max_d, seed)
    return jsonify({"ok": True})

@app.route("/api/settings/randomize", methods=["POST"])
def api_randomize_display():
    global _display_ids
    import random
    new_seed = random.randint(0, 999999)
    save_user_setting("display_seed", new_seed)
    max_d = get_user_setting("max_display", DEFAULT_MAX_DISPLAY)
    _display_ids = sample_display_ids(len(_issues), max_d, new_seed)
    return jsonify({"ok": True, "seed": new_seed, "count": len(_display_ids)})

@app.route("/api/reports")
def api_reports():
    analysis = build_report_analysis()
    return jsonify([{
        "id": r["id"], "title": r["title"],
        "matched_issues": r["matched_issues"],
        "matched_categories": r["matched_categories"],
    } for r in analysis])

@app.route("/api/reports/<report_id>")
def api_report_detail(report_id):
    analysis = build_report_analysis()
    for r in analysis:
        if r["id"] == report_id:
            highlighted = highlight_text(r["body"], _issues, r["matched_categories"])
            return jsonify({
                "id": r["id"], "title": r["title"],
                "body_html": highlighted,
                "matched_issues": r["matched_issues"],
                "matched_categories": r["matched_categories"],
            })
    return jsonify({"error": "not found"}), 404


@app.route("/api/preview_category", methods=["POST"])
def api_preview_category():
    """Like /api/search but uses multi-filter logic. Returns matched display-point scores."""
    data = request.get_json()
    filters = data.get("filters", [])
    if not filters:
        return jsonify({"scores": []})
    embs = get_active_embeddings()
    scores = apply_filter_logic(filters, _issues, embs, _display_ids)
    results = [{"id": int(k), "score": float(v)} for k, v in scores.items()]
    results.sort(key=lambda x: -x["score"])
    return jsonify({"scores": results})


@app.route("/api/cluster", methods=["POST"])
def api_cluster():
    """
    Cluster the display-subset embeddings using HDBSCAN, Agglomerative, or KMeans.

    Each cluster is represented by its *medoid* — the real issue text closest to the
    cluster centre — which becomes the semantic search query when added as a category.

    HDBSCAN  (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
    ─────────────────────────────────────────────────────────────────────────────────────
    Finds clusters as dense regions automatically; marks sparse outliers as noise (-1).
    Passes L2-normalised embeddings directly with metric='cosine' — do NOT use a
    precomputed distance matrix here, as sklearn's HDBSCAN MST construction degenerates
    badly with precomputed cosine matrices and collapses everything into 1-2 clusters.
    Key params:
      min_cluster_size  – smallest group that counts as a real cluster.
      min_samples       – density threshold; higher = fewer, tighter clusters + more noise.

    Agglomerative (bottom-up hierarchical)
    ───────────────────────────────────────
    Merges greedily from N singletons to K clusters; every point gets assigned.
    Uses precomputed cosine distance matrix with average or complete linkage.
    (Ward excluded: requires Euclidean geometry.)

    KMeans
    ──────
    Spherical KMeans on L2-normalised embeddings (= cosine KMeans). Fast and reliable
    when you know roughly how many clusters you want. Runs on the normed vectors
    directly so cosine similarity drives the centroid updates.
    Key param: k – number of clusters.

    Output (all algos)
    ──────────────────
    Clusters sorted by size descending. Medoid label used as the category query at a
    fixed default threshold when added.
    """
    data = request.get_json()
    algo = data.get("algo", "hdbscan")

    embs = get_active_embeddings()
    if embs is None:
        return jsonify({"error": "No embeddings loaded"}), 400

    disp_ids_arr = np.array(_display_ids, dtype=np.int64)
    disp_embs = embs[disp_ids_arr]
    n = len(disp_ids_arr)

    # L2-normalise: cosine similarity becomes inner product, cosine distance = 1 - sim
    norms = np.linalg.norm(disp_embs, axis=1, keepdims=True)
    normed = disp_embs / (norms + 1e-9)

    if algo == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN as _HDBSCAN
        except ImportError:
            return jsonify({"error": "scikit-learn >= 1.3 required for HDBSCAN"}), 500

        min_cluster_size = max(2, int(data.get("min_cluster_size", 15)))
        min_samples      = max(1, int(data.get("min_samples", 5)))

        # IMPORTANT: pass normed vectors with metric='cosine', NOT a precomputed matrix.
        # sklearn's HDBSCAN uses a minimum spanning tree internally; with precomputed
        # cosine distances the MST degenerates and collapses everything to 1-2 clusters.
        clusterer = _HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="cosine",
        )
        labels = clusterer.fit_predict(normed)

        # Build cosine dist matrix only for medoid computation (subset per cluster)
        sim_matrix = normed @ normed.T
        dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0).astype(np.float64)

    elif algo == "agglom":
        from sklearn.cluster import AgglomerativeClustering

        k       = max(2, min(100, int(data.get("k", 10))))
        linkage = data.get("linkage", "average")
        if linkage not in ("average", "complete"):
            linkage = "average"
        if k >= n:
            k = max(2, n // 2)

        sim_matrix = normed @ normed.T
        dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0).astype(np.float64)

        clusterer = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage=linkage,
        )
        labels = clusterer.fit_predict(dist_matrix)

    elif algo == "kmeans":
        from sklearn.cluster import KMeans

        k = max(2, min(100, int(data.get("k", 10))))
        if k >= n:
            k = max(2, n // 2)

        # Spherical KMeans: cluster normalised vectors, cosine similarity drives it
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(normed)

        sim_matrix = normed @ normed.T
        dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0).astype(np.float64)

    else:
        return jsonify({"error": f"Unknown algo: {algo}"}), 400

    # ── Build cluster list with medoid labels ───────────────────────────────────
    unique_labels = sorted(set(labels.tolist()) - {-1})  # exclude HDBSCAN noise
    clusters = []
    for lbl in unique_labels:
        member_local = np.where(labels == lbl)[0]
        # Medoid = member minimising total cosine distance to all cluster members
        sub_dist = dist_matrix[np.ix_(member_local, member_local)]
        medoid_local_idx = int(member_local[sub_dist.sum(axis=1).argmin()])
        orig_idx = int(disp_ids_arr[medoid_local_idx])
        clusters.append({
            "label": _issues[orig_idx],
            "size":  int(len(member_local)),
        })

    clusters.sort(key=lambda c: -c["size"])
    noise_count = int(np.sum(labels == -1))
    return jsonify({"clusters": clusters, "algo": algo, "noise": noise_count})


if __name__ == "__main__":
    initialise()
    app.run(debug=False, host="0.0.0.0", port=5000)
