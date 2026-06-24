"""
assistant/tools.py
──────────────────
Concrete implementations of the three required tools plus order storage.

Tools
-----
*  check_stock(sku)                       → stock level from live catalogue
*  create_order(dealer, items)            → structured order confirmation
*  find_parts_by_vehicle(vehicle_query)   → matching parts list

All tool functions return plain Python dicts — the agent layer converts
these to JSON strings before injecting them as tool results.

Design notes
------------
*  Orders are persisted to a JSON file (orders_db.json) at project root.
   In production this would be a database; for the assignment a flat file
   demonstrates structured-output validation without adding a DB dependency.
*  Pydantic v2 is used for order validation to guarantee the LLM cannot
   produce a malformed order payload.
*  check_stock re-reads the CSV each call so stock figures reflect any
   state changes made by create_order (units are decremented on order).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from assistant.retrieval import search_by_vehicle, get_by_sku

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent.parent
CATALOGUE_CSV = _HERE / "catalogue.csv"
ORDERS_DB     = _HERE / "orders_db.json"


# ── Pydantic models for structured output ─────────────────────────────────────
class OrderLineItem(BaseModel):
    sku:      str = Field(..., description="Product SKU code, e.g. BRK-1042")
    quantity: int = Field(..., gt=0, description="Number of units (must be > 0)")

    @field_validator("sku")
    @classmethod
    def normalise_sku(cls, v: str) -> str:
        return v.upper().strip()


class OrderRequest(BaseModel):
    dealer_name: str           = Field(..., min_length=1)
    items:       List[OrderLineItem]

    @field_validator("items")
    @classmethod
    def at_least_one_item(cls, v):
        if not v:
            raise ValueError("Order must contain at least one item.")
        return v


class OrderConfirmation(BaseModel):
    order_id:    str
    dealer_name: str
    timestamp:   str
    items:       List[Dict[str, Any]]   # enriched with name + unit_price
    total_inr:   int
    status:      str


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_catalogue() -> pd.DataFrame:
    return pd.read_csv(CATALOGUE_CSV)


def _load_orders() -> List[Dict]:
    if ORDERS_DB.exists():
        with open(ORDERS_DB, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_orders(orders: List[Dict]) -> None:
    with open(ORDERS_DB, "w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2, ensure_ascii=False)


# ── tool: check_stock ─────────────────────────────────────────────────────────
def check_stock(sku: str) -> Dict[str, Any]:
    """
    Return stock availability for a single SKU.

    Parameters
    ----------
    sku : str
        The product SKU code (case-insensitive).

    Returns
    -------
    dict with keys: sku, name, stock, status, price_inr
    """
    sku  = sku.upper().strip()
    item = get_by_sku(sku)

    if item is None:
        return {
            "success": False,
            "sku":     sku,
            "message": f"SKU '{sku}' not found in catalogue.",
        }

    stock  = int(item["stock"])
    status = "in_stock" if stock > 5 else ("low_stock" if stock > 0 else "out_of_stock")

    return {
        "success":   True,
        "sku":       item["sku"],
        "name":      item["name"],
        "stock":     stock,
        "status":    status,
        "price_inr": item["price_inr"],
    }


# ── tool: create_order ────────────────────────────────────────────────────────
def create_order(dealer_name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Place an order for a dealer. Validates input, checks stock, and persists.

    Parameters
    ----------
    dealer_name : str
    items : list of {"sku": str, "quantity": int}

    Returns
    -------
    OrderConfirmation dict on success, or error dict on failure.
    """
    # ── 1. Validate with Pydantic ─────────────────────────────────────────────
    try:
        req = OrderRequest(dealer_name=dealer_name, items=items)
    except Exception as exc:
        return {"success": False, "message": f"Validation error: {exc}"}

    cat = _load_catalogue()
    cat_idx = cat.set_index("sku")

    enriched_items = []
    total          = 0
    errors         = []

    # ── 2. Validate each line item ────────────────────────────────────────────
    for line in req.items:
        if line.sku not in cat_idx.index:
            errors.append(f"SKU '{line.sku}' not found.")
            continue

        row       = cat_idx.loc[line.sku]
        available = int(row["stock"])

        if available < line.quantity:
            errors.append(
                f"Insufficient stock for {line.sku}: requested {line.quantity}, "
                f"available {available}."
            )
            continue

        unit_price = int(row["price_inr"])
        line_total = unit_price * line.quantity
        total     += line_total

        enriched_items.append({
            "sku":        line.sku,
            "name":       str(row["name"]),
            "quantity":   line.quantity,
            "unit_price": unit_price,
            "line_total": line_total,
        })

    if errors:
        return {"success": False, "message": " | ".join(errors)}

    # ── 3. Deduct stock ───────────────────────────────────────────────────────
    for line in enriched_items:
        cat.loc[cat["sku"] == line["sku"], "stock"] -= line["quantity"]
    cat.to_csv(CATALOGUE_CSV, index=False)

    # ── 4. Persist order ──────────────────────────────────────────────────────
    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    ts       = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    confirmation = OrderConfirmation(
        order_id    = order_id,
        dealer_name = req.dealer_name,
        timestamp   = ts,
        items       = enriched_items,
        total_inr   = total,
        status      = "confirmed",
    ).model_dump()

    orders = _load_orders()
    orders.append(confirmation)
    _save_orders(orders)

    # Update retrieval cache so get_by_sku and search see updated stock
    from assistant import retrieval as _r
    _r._meta = cat.to_dict(orient="records")

    return {"success": True, **confirmation}


# ── tool: find_parts_by_vehicle ───────────────────────────────────────────────
def find_parts_by_vehicle(vehicle_query: str, top_k: int = 8) -> Dict[str, Any]:
    """
    Find catalogue parts that fit a given vehicle.

    Parameters
    ----------
    vehicle_query : str
        Natural language vehicle description, e.g. "KTM Duke 390" or "Pulsar 150".

    Returns
    -------
    dict with keys: success, vehicle_query, count, parts (list of product dicts)
    """
    results = search_by_vehicle(vehicle_query, top_k=top_k)

    if not results:
        return {
            "success":       False,
            "vehicle_query": vehicle_query,
            "message":       f"No parts found for '{vehicle_query}'.",
        }

    parts = [
        {
            "sku":             r["sku"],
            "name":            r["name"],
            "category":        r["category"],
            "brand":           r["brand"],
            "vehicle_fitment": r["vehicle_fitment"],
            "price_inr":       r["price_inr"],
            "stock":           r["stock"],
        }
        for r in results
    ]

    return {
        "success":       True,
        "vehicle_query": vehicle_query,
        "count":         len(parts),
        "parts":         parts,
    }
