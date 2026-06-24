"""
assistant/retrieval.py
─────────────────────
RAG pipeline for the VIKMO product catalogue.

Strategy
--------
1.  Load catalogue.csv (600 SKUs).
2.  Build a rich text representation per SKU that combines name, category,
    brand, vehicle fitment, price, stock level, and description.
3.  Embed each text with `sentence-transformers/all-MiniLM-L6-v2`
    (small, fast, free — no API key needed).
4.  Store vectors in a flat FAISS index (IndexFlatIP with normalised
    vectors == cosine similarity).
5.  Persist the index + metadata to disk so the first expensive build is
    cached across restarts.

Design decisions
----------------
*  all-MiniLM-L6-v2 was chosen because it is purpose-built for semantic
   similarity, fits in <100 MB, runs on CPU in <1 s per batch, and
   outperforms BM25 on product-search benchmarks at this scale.
*  Cosine similarity (inner product on L2-normalised vectors) is better
   than Euclidean distance for text retrieval.
*  The rich text repr deliberately includes vehicle fitment verbatim so
   "brake pad for Pulsar 150" queries align with that field without
   needing a separate sparse index.
*  We use a flat (brute-force) index because 600 vectors fit in RAM with
   room to spare and exact search avoids quantisation artefacts.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ── paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.parent          # project root
CATALOGUE_CSV = _HERE / "catalogue.csv"
INDEX_CACHE   = _HERE / ".cache" / "faiss.index"
META_CACHE    = _HERE / ".cache" / "metadata.pkl"

# ── model ────────────────────────────────────────────────────────────────────
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ── text representation ───────────────────────────────────────────────────────
def _make_doc(row: pd.Series) -> str:
    """Combine catalogue fields into a single searchable string."""
    return (
        f"SKU: {row['sku']}. "
        f"Name: {row['name']}. "
        f"Category: {row['category']}. "
        f"Brand: {row['brand']}. "
        f"Fits: {row['vehicle_fitment']}. "
        f"Price: ₹{row['price_inr']}. "
        f"Stock: {row['stock']} units. "
        f"Description: {row['description']}."
    )


# ── index build / load ────────────────────────────────────────────────────────
def _build_index(df: pd.DataFrame):
    """Embed all catalogue rows and return (faiss_index, metadata_list)."""
    model  = _get_model()
    docs   = [_make_doc(r) for _, r in df.iterrows()]
    meta   = df.to_dict(orient="records")

    print(f"[RAG] Encoding {len(docs)} catalogue entries …")
    vecs = model.encode(docs, batch_size=64, show_progress_bar=True,
                        normalize_embeddings=True)
    vecs = np.array(vecs, dtype="float32")

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product == cosine on normalised vecs
    index.add(vecs)

    INDEX_CACHE.parent.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_CACHE))
    with open(META_CACHE, "wb") as f:
        pickle.dump(meta, f)

    print(f"[RAG] Index saved to {INDEX_CACHE}")
    return index, meta


def _load_index():
    index = faiss.read_index(str(INDEX_CACHE))
    with open(META_CACHE, "rb") as f:
        meta = pickle.load(f)
    return index, meta


# ── public API ────────────────────────────────────────────────────────────────
_index: faiss.Index | None = None
_meta:  List[Dict]         = []


def init_retrieval(force_rebuild: bool = False) -> None:
    """Call once at startup to load (or build) the FAISS index."""
    global _index, _meta
    if not force_rebuild and INDEX_CACHE.exists() and META_CACHE.exists():
        print("[RAG] Loading cached index …")
        _index, _meta = _load_index()
        print(f"[RAG] Loaded {_index.ntotal} vectors.")
    else:
        df = pd.read_csv(CATALOGUE_CSV)
        _index, _meta = _build_index(df)


def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Embed *query* and return the top-k most similar catalogue entries.

    Returns a list of dicts (catalogue row fields).
    """
    if _index is None:
        raise RuntimeError("Call init_retrieval() before search().")

    model = _get_model()
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(q_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = dict(_meta[idx])
        item["_similarity"] = float(score)
        results.append(item)
    return results


def search_by_vehicle(vehicle: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Filter-first retrieval: exact substring match on vehicle_fitment,
    then rank by embedding similarity as a secondary signal.
    """
    # Load full catalogue for filtering
    df = pd.read_csv(CATALOGUE_CSV)
    mask = df["vehicle_fitment"].str.lower().str.contains(
        vehicle.lower(), na=False
    )
    filtered = df[mask | (df["vehicle_fitment"].str.lower() == "universal")]

    if filtered.empty:
        # Fallback to pure semantic search
        return search(f"parts for {vehicle}", top_k=top_k)

    # Rank filtered results by semantic similarity to the vehicle query
    model = _get_model()
    docs  = [_make_doc(r) for _, r in filtered.iterrows()]
    vecs  = model.encode(docs, normalize_embeddings=True).astype("float32")
    q_vec = model.encode([f"parts for {vehicle}"],
                          normalize_embeddings=True).astype("float32")
    sims  = (vecs @ q_vec.T).squeeze()
    order = np.argsort(-sims)[:top_k]

    results = []
    for i in order:
        item = filtered.iloc[i].to_dict()
        item["_similarity"] = float(sims[i])
        results.append(item)
    return results


def get_by_sku(sku: str) -> Dict[str, Any] | None:
    """Direct SKU lookup (O(n) scan of metadata list)."""
    sku = sku.upper().strip()
    for item in _meta:
        if item["sku"].upper() == sku:
            return dict(item)
    return None
