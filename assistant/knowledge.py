"""
What Vaayu knows, searchable by meaning - kept only on this laptop.

One small store per kind, in data/knowledge/:
  facts    about the user (memory.py writes these, Mem0-style)
  notes    things he read about while idle (research.py), with the source URL
  answers  spoken answers to web questions, for instant repeats (web_answer.py)
  lessons  what the daily review learned from the logs

Embeddings: BAAI/bge-small-en-v1.5 via fastembed - 384 numbers per text, on
the CPU (about 130 MB, no GPU memory), cached under models/fastembed. Each
kind is a .jsonl of rows plus a .npy matrix (float16); search is one numpy
dot product. Nothing here may raise into the caller's spoken reply path
unless it's a plain bug - callers wrap it.
"""

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
DIR = _ROOT / "data" / "knowledge"
MODEL_CACHE = _ROOT / "models" / "fastembed"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_lock = threading.RLock()
_model = None
_cache: dict[Path, tuple[list[dict], np.ndarray]] = {}


def _embed(texts: list[str]) -> np.ndarray:
    """Unit-length float32 vectors, one row per text."""
    global _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding
            _model = TextEmbedding(EMBED_MODEL, cache_dir=str(MODEL_CACHE))
        vecs = np.array(list(_model.embed(texts)), dtype=np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


def warm_up() -> None:
    """Load the embedding model in the background - the first load took ~16 s live."""
    def run():
        try:
            _embed(["warm up"])
        except Exception:
            pass  # no model (offline first run): callers fall back on their own
    threading.Thread(target=run, daemon=True, name="embed-warmup").start()


def _paths(kind: str) -> tuple[Path, Path]:
    return DIR / f"{kind}.jsonl", DIR / f"{kind}.npy"


def _load(kind: str) -> tuple[list[dict], np.ndarray]:
    rows_path, vec_path = _paths(kind)
    if rows_path in _cache:
        return _cache[rows_path]
    rows, mat = [], np.zeros((0, 384), np.float32)
    try:
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mat = np.load(vec_path).astype(np.float32)
    except (OSError, ValueError):
        pass
    if len(rows) != len(mat):  # a half-written pair: rebuild the vectors from the text
        mat = _embed([r["text"] for r in rows]) if rows else np.zeros((0, 384), np.float32)
    _cache[rows_path] = (rows, mat)
    return rows, mat


def _write(kind: str, rows: list[dict], mat: np.ndarray) -> None:
    # ponytail: rewrites the whole kind on every change - fine for thousands of rows; append-only
    # files (or LanceDB on disk) if a kind ever reaches hundreds of thousands
    rows_path, vec_path = _paths(kind)
    DIR.mkdir(parents=True, exist_ok=True)
    rows_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    with open(vec_path, "wb") as f:
        np.save(f, mat.astype(np.float16))
    _cache[rows_path] = (rows, mat)


def all(kind: str) -> list[dict]:  # noqa: A001 - the module's own vocabulary: knowledge.all("facts")
    with _lock:
        return list(_load(kind)[0])


def add(kind: str, text: str, meta: dict | None = None, dedupe: float = 0.92) -> tuple[dict, bool]:
    """Store text; if something at least `dedupe` similar is already there, update that row
    instead. Returns (row, updated)."""
    text = text.strip()
    vec = _embed([text])[0]
    now = datetime.now().isoformat(timespec="seconds")
    with _lock:
        rows, mat = _load(kind)
        if len(rows):
            scores = mat @ vec
            i = int(np.argmax(scores))
            if scores[i] >= dedupe:
                rows = list(rows)
                rows[i] = {**rows[i], "text": text, "meta": {**rows[i].get("meta", {}), **(meta or {})}, "ts": now}
                mat = mat.copy()
                mat[i] = vec
                _write(kind, rows, mat)
                return rows[i], True
        row = {"id": uuid.uuid4().hex[:12], "text": text, "meta": meta or {}, "ts": now}
        _write(kind, rows + [row], np.vstack([mat, vec[None, :]]))
        return row, False


def replace(kind: str, row_id: str, text: str) -> bool:
    """Rewrite one row's text (and its vector) - e.g. a fact the user corrected."""
    vec = _embed([text.strip()])[0]
    with _lock:
        rows, mat = _load(kind)
        for i, r in enumerate(rows):
            if r["id"] == row_id:
                rows, mat = list(rows), mat.copy()
                rows[i] = {**r, "text": text.strip(), "ts": datetime.now().isoformat(timespec="seconds")}
                mat[i] = vec
                _write(kind, rows, mat)
                return True
    return False


def search(kind: str, query: str, k: int = 5, min_score: float = 0.5) -> list[tuple[float, dict]]:
    if not query or not query.strip():
        return []
    with _lock:
        rows, mat = _load(kind)
        if not rows:
            return []
        scores = mat @ _embed([query])[0]
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), rows[i]) for i in top if scores[i] >= min_score]


def remove(kind: str, ids) -> int:
    ids = set(ids)
    with _lock:
        rows, mat = _load(kind)
        keep = [i for i, r in enumerate(rows) if r["id"] not in ids]
        if len(keep) == len(rows):
            return 0
        _write(kind, [rows[i] for i in keep], mat[keep] if keep else np.zeros((0, mat.shape[1]), np.float32))
        return len(rows) - len(keep)


def consolidate(kind: str, near: float = 0.92, max_age_days: float | None = None) -> tuple[int, int]:
    """Build plan's weekly consolidation: near-duplicates merged (the newest wording kept) and rows older
    than max_age_days dropped. Returns (merged, dropped)."""
    with _lock:
        rows, mat = _load(kind)
        keep, merged, dropped = [], 0, 0
        # newest first; the same second: the later row is the newer one
        for i in sorted(range(len(rows)), key=lambda i: (rows[i].get("ts", ""), i), reverse=True):
            if max_age_days is not None and age_hours(rows[i]) > max_age_days * 24:
                dropped += 1
            elif keep and float(np.max(mat[keep] @ mat[i])) >= near:
                merged += 1
            else:
                keep.append(i)
        if merged or dropped:
            keep.sort()
            _write(kind, [rows[i] for i in keep], mat[keep] if keep else np.zeros((0, mat.shape[1]), np.float32))
        return merged, dropped


def age_hours(row: dict) -> float:
    try:
        return (time.time() - datetime.fromisoformat(row["ts"]).timestamp()) / 3600
    except (KeyError, ValueError):
        return float("inf")
