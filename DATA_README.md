# VIKMO — AI / ML Intern Assignment

This bundle contains your assignment brief and the data you need.

| File | Purpose |
|---|---|
| `VIKMO_AI_ML_Intern_Assignment.pdf` | The assignment brief. Start here. |
| `catalogue.csv` / `catalogue.json` | The product catalogue — your retrieval corpus for **Part A** (the assistant). Same data, two formats. |
| `sales_history.csv` | Weekly sales for the **Part B** forecasting bonus. |

All prices are in **INR (₹)**, integers.

## `catalogue.csv` / `catalogue.json` — product catalogue (600 SKUs)

| Field | Type | Notes |
|---|---|---|
| `sku` | string | Unique product code (e.g. `BRK-1042`). |
| `name` | string | Display name, e.g. `Brake Pad Set — Bajaj Pulsar 150`. |
| `category` | string | One of 11 categories. |
| `brand` | string | Part brand. |
| `vehicle_fitment` | string | Make + model (e.g. `KTM Duke 390`) or `Universal`. Use this for `find_parts_by_vehicle`. |
| `price_inr` | integer | Selling price in ₹. |
| `stock` | integer | Units on hand (use for `check_stock`). Some SKUs are 0 or single-digit. |
| `description` | string | Short product blurb. |

The catalogue is large enough that stuffing the whole thing into the prompt is not a valid approach — implement real retrieval.

## `sales_history.csv` — weekly sales for forecasting (2,340 rows)

30 SKUs × 78 weekly observations (Mondays, **2024-12-16 → 2026-06-08**). Series include a per-SKU base level, mild trend, yearly seasonality, a festive lift (Oct–Nov), promotional spikes, and Poisson noise. The most recent weeks are suitable as a held-out test window.

| Field | Type | Notes |
|---|---|---|
| `date` | ISO date | Week start (Monday). |
| `sku` | string | Matches a `sku` in the catalogue. |
| `units_sold` | integer | Units sold that week. |
| `promo_flag` | 0/1 | 1 if that week had a promotion (expect elevated demand). |
