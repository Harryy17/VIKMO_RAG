"""
forecasting/forecast.py
───────────────────────
Part B — Demand Forecasting for VIKMO auto parts.

Approach
--------
We forecast weekly unit sales for the next 4 weeks (held-out test window)
for each of the 30 SKUs in sales_history.csv.

Models implemented
------------------
1.  Naive Last Value        — baseline: last observed value repeated.
2.  Seasonal Naive (4-week) — baseline: value from 4 weeks ago.
3.  Rolling 4-week MA       — baseline: mean of last 4 weeks.
4.  Exponential Smoothing   — Holt-Winters with additive seasonality
                              (our primary model).
5.  SARIMA                  — ARIMA(1,1,1)(1,0,1,4) per SKU (optional,
                              activated with --sarima flag).

Validation scheme (no leakage)
-------------------------------
*  Test window  : last 4 weeks of the series (2026-05-19 → 2026-06-08).
*  Training data: everything before the test window.
*  Models are fitted ONLY on training data; test data is never seen during
   training or hyperparameter selection.
*  Metrics (MAE, MAPE) are computed on the held-out test window.

Metrics
-------
*  MAE  (Mean Absolute Error)           — absolute scale, easy to interpret.
*  MAPE (Mean Absolute Percentage Error) — scale-free, but undefined when
   true value is 0 (we skip those weeks for MAPE).

Run:
    python forecasting/forecast.py

With SARIMA (slow):
    python forecasting/forecast.py --sarima

Output:
    forecasting/results.csv   — per-SKU per-week predictions vs actuals
    forecasting/summary.csv   — per-model MAE and MAPE summary
    forecasting/plots/        — per-SKU forecast plots (optional)
"""

from __future__ import annotations

import argparse
import sys
import warnings

# Fix Windows console encoding for Unicode box-drawing chars
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parent.parent
SALES_CSV    = _ROOT / "sales_history.csv"
OUT_DIR      = Path(__file__).parent
RESULTS_CSV  = OUT_DIR / "results.csv"
SUMMARY_CSV  = OUT_DIR / "summary.csv"

# ── constants ─────────────────────────────────────────────────────────────────
N_TEST_WEEKS = 4   # hold-out size


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    df = pd.read_csv(SALES_CSV, parse_dates=["date"])
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)
    return df


def train_test_split_sku(
    sku_df: pd.DataFrame, n_test: int = N_TEST_WEEKS
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a single-SKU series into train / test by last n_test rows."""
    return sku_df.iloc[:-n_test], sku_df.iloc[-n_test:]


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline models
# ═══════════════════════════════════════════════════════════════════════════════

def naive_last_value(train: pd.Series, n_steps: int) -> np.ndarray:
    """Forecast = last observed value, repeated."""
    return np.full(n_steps, train.iloc[-1])


def seasonal_naive(train: pd.Series, n_steps: int, season: int = 4) -> np.ndarray:
    """Forecast = value from `season` periods ago."""
    preds = []
    series = list(train)
    for i in range(n_steps):
        idx = -(season - (i % season))
        preds.append(series[idx])
    return np.array(preds)


def rolling_ma(train: pd.Series, n_steps: int, window: int = 4) -> np.ndarray:
    """Forecast = mean of last `window` observations."""
    avg = train.iloc[-window:].mean()
    return np.full(n_steps, avg)


# ═══════════════════════════════════════════════════════════════════════════════
# Primary model — Holt-Winters Exponential Smoothing
# ═══════════════════════════════════════════════════════════════════════════════

def holtwinters_forecast(train: pd.Series, n_steps: int) -> np.ndarray:
    """
    Holt-Winters with additive seasonality (period = 4 weeks).

    If the series is too short for seasonal fitting (< 2 * 4 = 8 obs),
    fall back to simple exponential smoothing.
    """
    try:
        if len(train) >= 2 * 4:
            model = ExponentialSmoothing(
                train,
                trend          = "add",
                seasonal       = "add",
                seasonal_periods = 4,
                initialization_method = "estimated",
            ).fit(optimized=True, disp=False)
        else:
            model = ExponentialSmoothing(
                train,
                trend          = "add",
                initialization_method = "estimated",
            ).fit(optimized=True, disp=False)

        preds = model.forecast(n_steps)
        return np.maximum(preds.values, 0)   # demand can't be negative

    except Exception:
        # Absolute fallback: rolling MA
        return rolling_ma(train, n_steps)


# ═══════════════════════════════════════════════════════════════════════════════
# Optional: SARIMA
# ═══════════════════════════════════════════════════════════════════════════════

def sarima_forecast(train: pd.Series, n_steps: int) -> np.ndarray:
    """ARIMA(1,1,1)(1,0,1,4) — slower but captures autocorrelation."""
    try:
        model = SARIMAX(
            train,
            order          = (1, 1, 1),
            seasonal_order = (1, 0, 1, 4),
            enforce_stationarity  = False,
            enforce_invertibility = False,
        ).fit(disp=False)
        preds = model.forecast(n_steps)
        return np.maximum(preds.values, 0)
    except Exception:
        return holtwinters_forecast(train, n_steps)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_forecast(use_sarima: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all models on all SKUs. Returns (results_df, summary_df).

    results_df : long-format with columns
        [sku, date, actual, naive_lv, seasonal_naive, rolling_ma,
         holt_winters, (sarima)]

    summary_df : model-level MAE and MAPE aggregated across all SKUs.
    """
    df   = load_data()
    skus = df["sku"].unique()

    all_rows   = []
    sku_scores = {
        "naive_lv":        [],
        "seasonal_naive":  [],
        "rolling_ma":      [],
        "holt_winters":    [],
    }
    if use_sarima:
        sku_scores["sarima"] = []

    print(f"Forecasting {len(skus)} SKUs for {N_TEST_WEEKS} weeks ahead …")

    for sku in skus:
        sub   = df[df["sku"] == sku].copy()
        train_df, test_df = train_test_split_sku(sub)

        y_train  = train_df["units_sold"].values.astype(float)
        y_test   = test_df["units_sold"].values.astype(float)
        dates    = test_df["date"].values

        s_train  = pd.Series(y_train)

        preds = {
            "naive_lv":       naive_last_value(s_train, N_TEST_WEEKS),
            "seasonal_naive": seasonal_naive(s_train,   N_TEST_WEEKS),
            "rolling_ma":     rolling_ma(s_train,       N_TEST_WEEKS),
            "holt_winters":   holtwinters_forecast(s_train, N_TEST_WEEKS),
        }
        if use_sarima:
            preds["sarima"] = sarima_forecast(s_train, N_TEST_WEEKS)

        # Accumulate per-SKU metrics
        for model_name, pred in preds.items():
            sku_scores[model_name].append({
                "sku":  sku,
                "mae":  mae(y_test, pred),
                "mape": mape(y_test, pred),
            })

        # Build results rows
        for i, (date, actual) in enumerate(zip(dates, y_test)):
            row = {"sku": sku, "date": date, "actual": int(actual)}
            for model_name, pred in preds.items():
                row[model_name] = round(float(pred[i]), 2)
            all_rows.append(row)

    results_df = pd.DataFrame(all_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_rows = []
    for model_name, scores in sku_scores.items():
        sc_df     = pd.DataFrame(scores)
        avg_mae   = sc_df["mae"].mean()
        avg_mape  = sc_df["mape"].dropna().mean()
        summary_rows.append({
            "model":    model_name,
            "avg_mae":  round(avg_mae,  2),
            "avg_mape": round(avg_mape, 2),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("avg_mae")
    return results_df, summary_df


def print_summary(summary_df: pd.DataFrame) -> None:
    SEP = "=" * 55
    print("\n" + SEP)
    print("  DEMAND FORECASTING RESULTS")
    print(SEP)
    print(f"  {'Model':<20} {'Avg MAE':>10} {'Avg MAPE':>12}")
    print("-" * 55)
    for _, row in summary_df.iterrows():
        marker = " <- best" if row["model"] == summary_df.iloc[0]["model"] else ""
        print(
            f"  {row['model']:<20} {row['avg_mae']:>10.2f} "
            f"{row['avg_mape']:>11.1f}%{marker}"
        )
    print(SEP)

    # Does best model beat all baselines?
    best      = summary_df.iloc[0]["model"]
    baselines = ["naive_lv", "seasonal_naive", "rolling_ma"]
    if best not in baselines:
        best_mae           = summary_df.iloc[0]["avg_mae"]
        baseline_maes      = summary_df[summary_df["model"].isin(baselines)]["avg_mae"]
        worst_baseline_mae = baseline_maes.max()
        print(
            f"\n  [OK] {best} (MAE={best_mae:.2f}) beats all baselines "
            f"(worst baseline MAE={worst_baseline_mae:.2f})"
        )
    else:
        print(f"\n  [WARN] A simple baseline won. The data may be noisy or very seasonal.")


def main():
    parser = argparse.ArgumentParser(description="VIKMO Demand Forecasting")
    parser.add_argument("--sarima", action="store_true",
                        help="Also fit SARIMA (slower)")
    args = parser.parse_args()

    results_df, summary_df = run_forecast(use_sarima=args.sarima)

    results_df.to_csv(RESULTS_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    print_summary(summary_df)
    print(f"\n  Detailed results → {RESULTS_CSV}")
    print(f"  Summary          → {SUMMARY_CSV}\n")


if __name__ == "__main__":
    main()
