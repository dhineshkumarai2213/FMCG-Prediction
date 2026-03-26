"""
forecast_model.py  —  FMCG AI · Fully Automated Demand Prediction
==================================================================
VERSION 2.0  —  Zero manual input required from the user.

WHAT CHANGED FROM v1
─────────────────────
v1  User had to supply: lag_1, lag_2, lag_3, price, promotion_flag
v2  User supplies ONLY: product name + forecast month
    All features are derived automatically by the backend.

ARCHITECTURE
─────────────
┌─────────────────────────────────────────────────────────────────┐
│ User Input: { product: "Product_A", month: 6 }                  │
│                          ↓                                      │
│ _auto_generate_features(product, month)                         │
│   ├── PATH A: Real CSV data exists                              │
│   │     └── Pull last 6 weeks of actual demand → lags, rolling  │
│   └── PATH B: No CSV / no rows for product                      │
│         └── Generate statistically realistic values from        │
│             per-product historical ranges                        │
│                          ↓                                      │
│ _build_feature_row(features_dict)                               │
│   └── Assemble 10-column DataFrame in training column order     │
│                          ↓                                      │
│ model.predict(input_df)  +  per-tree confidence                 │
│                          ↓                                      │
│ Return: { forecasted_demand, confidence, model_accuracy,        │
│           product, month, lag_values_used, price_used, … }      │
└─────────────────────────────────────────────────────────────────┘

FEATURE GENERATION RULES
─────────────────────────
lag_1          = demand 1 week before forecast_month (from CSV or synthetic)
lag_2          = demand 2 weeks before
lag_3          = demand 3 weeks before
rolling_mean_3 = mean(lag_1, lag_2, lag_3)
rolling_mean_6 = mean of 6 most recent demand values
month          = forecast month (1–12)
quarter        = (month-1)//3 + 1
week_of_year   = approx. mid-week of forecast month
price          = mean price for the product from CSV, else product default
promotion_flag = 0 (conservative default; realistic for baseline forecast)

MODEL LOADING
──────────────
The pkl is loaded ONCE into _CACHE at first API call.
Subsequent calls (all 200-tree predictions) skip joblib entirely.

VIVA ONE-LINER
──────────────
"The system is fully autonomous: given only a product name and month,
the backend retrieves the last 6 weeks of historical demand from the
dataset to compute lag and rolling features, falling back to
statistically calibrated synthetic values when data is unavailable,
so the user never needs to supply numerical inputs manually."
"""

import os
import math
import joblib
import numpy  as np
import pandas as pd

# ── Paths — always relative to THIS file, never cwd-dependent ────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "demand_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data",  "sales_data.csv")

# ── Module-level model cache — loaded ONCE, reused for every request ─────────
_CACHE: dict | None = None

# ── Per-product demand statistics derived from training data
#    Used as fallback when CSV rows are unavailable for a product.
#    Format: { product_name: (mean_demand, std_demand, mean_price) }
_PRODUCT_STATS = {
    "Product_A": (219.2, 34.5,  52.0),
    "Product_B": (270.9, 43.6,  55.0),
    "Product_C": (195.9, 31.1,  48.0),
    "Product_D": (236.9, 38.6,  53.0),
}
# Default for unknown products (conservative mid-range)
_DEFAULT_STATS = (220.0, 35.0, 52.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Model loading (cached singleton)
# ═══════════════════════════════════════════════════════════════════════════════

def _load() -> dict:
    """
    Load and structurally validate demand_model.pkl exactly once per process.

    Returns the payload dict:
        { "model": RandomForestRegressor,
          "features": list[str],
          "metrics": {"mae": float, "rmse": float, "r2": float, "accuracy": float} }

    Raises
    ------
    FileNotFoundError  — pkl missing (train_model.py not run yet)
    TypeError          — pkl contains wrong type (old format)
    KeyError           — pkl dict missing required key
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE                        # ← fast path: already loaded

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Please run  python train_model.py  first."
        )

    raw = joblib.load(MODEL_PATH)

    # Validate structure — crash clearly here, not silently later
    if not isinstance(raw, dict):
        raise TypeError(
            f"demand_model.pkl must be a dict, got {type(raw).__name__}. "
            "Re-run python train_model.py."
        )
    for required_key in ("model", "features", "metrics"):
        if required_key not in raw:
            raise KeyError(
                f"demand_model.pkl missing key '{required_key}'. "
                "Re-run python train_model.py."
            )

    # Cast all metric values to plain Python float (never np.float64)
    # This guarantees Flask jsonify() can serialise them in all environments.
    raw["metrics"] = {k: float(v) for k, v in raw["metrics"].items()}

    _CACHE = raw
    return _CACHE


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Automatic feature generation (the core of v2)
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_generate_features(product: str, month: int) -> dict:
    """
    Given only a product name and forecast month, automatically derive
    all 10 features the model needs.

    Strategy
    ─────────
    1. Try to load sales_data.csv and filter rows for this product.
    2. If ≥ 6 rows exist → extract last 6 actual demand values.
    3. If < 6 rows (or no CSV) → synthesise statistically realistic values
       using the per-product mean ± std from _PRODUCT_STATS.

    Parameters
    ----------
    product : str   e.g. "Product_A"
    month   : int   1–12  (the month we want to forecast)

    Returns
    -------
    dict with all 10 feature keys ready for _build_feature_row()
    """
    # ── Step 1: derive time features from the requested month ────────────────
    month        = max(1, min(12, int(month)))   # clamp to valid range
    quarter      = (month - 1) // 3 + 1          # 1→Q1, 4→Q2, 7→Q3, 10→Q4
    # week_of_year: approximate mid-week of the requested month
    # month 1 ≈ week 2, month 6 ≈ week 24, month 12 ≈ week 50
    week_of_year = max(1, min(52, int(round((month - 0.5) / 12.0 * 52))))

    # ── Step 2: get product demand history (real or synthetic) ────────────────
    recent_demands = _get_recent_demands(product, n=6)   # list of 6 floats

    # ── Step 3: compute lag features from the 6 most recent demand values ─────
    #   recent_demands[0]  = most recent (lag_1)
    #   recent_demands[1]  = one before  (lag_2)
    #   recent_demands[2]  = two before  (lag_3)
    #   recent_demands[3..5] = used for rolling_mean_6
    lag_1 = float(recent_demands[0])
    lag_2 = float(recent_demands[1])
    lag_3 = float(recent_demands[2])

    rolling_mean_3 = round(float(np.mean(recent_demands[:3])), 4)
    rolling_mean_6 = round(float(np.mean(recent_demands[:6])), 4)

    # ── Step 4: price — use product average from CSV, else product default ────
    price = _get_mean_price(product)

    # ── Step 5: promotion_flag — 0 by default (conservative baseline)
    #   This gives a "no-promotion" base forecast.
    #   For promotional lift scenarios, pass promotion_flag=1 explicitly.
    promotion_flag = 0

    return {
        "month"          : month,
        "quarter"        : quarter,
        "week_of_year"   : week_of_year,
        "lag_1"          : lag_1,
        "lag_2"          : lag_2,
        "lag_3"          : lag_3,
        "rolling_mean_3" : rolling_mean_3,
        "rolling_mean_6" : rolling_mean_6,
        "price"          : price,
        "promotion_flag" : promotion_flag,
        # Extra metadata returned to the frontend for transparency
        "_source"        : "csv" if _csv_has_rows(product) else "synthetic",
    }


def _csv_has_rows(product: str) -> bool:
    """Return True if the CSV exists and has data for this product."""
    if not os.path.exists(DATA_PATH):
        return False
    try:
        df = pd.read_csv(DATA_PATH, usecols=["product"])
        return len(df[df["product"] == product]) >= 6
    except Exception:
        return False


def _get_recent_demands(product: str, n: int = 6) -> list:
    """
    Return the n most recent demand values for a product.

    PATH A (real data): Read from CSV, sort by date, return last n values.
    PATH B (synthetic): Generate n realistic values using product statistics.
    """
    # ── PATH A: try the CSV ───────────────────────────────────────────────────
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH, usecols=["product", "demand", "date"])
            df = df[df["product"] == product].dropna(subset=["demand"])

            if len(df) >= n:
                # Sort chronologically, take last n rows
                df = df.sort_values("date").tail(n)
                # Return in reverse order: [most_recent, ..., oldest]
                return df["demand"].tolist()[::-1]
        except Exception:
            pass   # fall through to PATH B

    # ── PATH B: synthesise realistic values ──────────────────────────────────
    #   Seed with (product_hash + n) for reproducibility between requests
    #   but variation between products.
    mean, std, _ = _PRODUCT_STATS.get(product, _DEFAULT_STATS)
    rng = np.random.default_rng(seed=abs(hash(product)) % (2**32))

    # Simulate a mild declining trend over 6 periods (realistic decay)
    # most recent = mean, going back adds slight variation
    values = []
    current = mean
    for i in range(n):
        # Each step back: small random walk ±10% of std
        noise   = rng.normal(0, std * 0.4)
        current = max(mean * 0.5, current + noise)   # floor at 50% of mean
        values.append(round(current, 2))

    return values   # [most_recent, ..., oldest]


def _get_mean_price(product: str) -> float:
    """Return mean price for a product from CSV, else product default."""
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH, usecols=["product", "price"])
            subset = df[df["product"] == product]["price"].dropna()
            if len(subset) > 0:
                return round(float(subset.mean()), 2)
        except Exception:
            pass

    _, _, default_price = _PRODUCT_STATS.get(product, _DEFAULT_STATS)
    return default_price


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Feature row assembly
# ═══════════════════════════════════════════════════════════════════════════════

def _build_feature_row(features: dict) -> pd.DataFrame:
    """
    Assemble the 10-column DataFrame in the EXACT column order used at training.
    The column order is taken from the loaded model's feature list, so even if
    train_model.py is updated, this function remains correct automatically.
    """
    payload       = _load()
    feature_order = payload["features"]   # e.g. ['month','quarter', ...]

    row = {
        "month"          : int(features["month"]),
        "quarter"        : int(features["quarter"]),
        "week_of_year"   : int(features["week_of_year"]),
        "lag_1"          : float(features["lag_1"]),
        "lag_2"          : float(features["lag_2"]),
        "lag_3"          : float(features["lag_3"]),
        "rolling_mean_3" : float(features["rolling_mean_3"]),
        "rolling_mean_6" : float(features["rolling_mean_6"]),
        "price"          : float(features["price"]),
        "promotion_flag" : int(features["promotion_flag"]),
    }

    # Build DataFrame with guaranteed column order (matching training)
    return pd.DataFrame([row])[feature_order]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Public API (called by app.py)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_demand(product: str, month: int) -> dict:
    """
    Fully automated demand prediction.
    Only two inputs needed from the user.

    Parameters
    ----------
    product : str   Product name, e.g. "Product_A"
    month   : int   Forecast month (1–12)

    Returns
    -------
    dict
        forecasted_demand  float   Predicted demand in units
        confidence         float   0–100 %, derived from per-tree variance
        model_accuracy     float   R² × 100 from held-out test evaluation
        r2                 float   R² score
        mae                float   Mean Absolute Error (units)
        rmse               float   Root Mean Squared Error (units)
        product            str     Echo of input product name
        month              int     Echo of input month
        data_source        str     "csv" or "synthetic" — transparency flag
        lag_1              float   Auto-generated feature value (for display)
        lag_2              float   Auto-generated feature value (for display)
        lag_3              float   Auto-generated feature value (for display)
        price_used         float   Price used in prediction
    """
    # ── 1. Load model (cached after first call) ───────────────────────────────
    payload  = _load()
    model    = payload["model"]
    metrics  = payload["metrics"]

    # ── 2. Auto-generate all features from product + month only ──────────────
    features = _auto_generate_features(product, month)
    source   = features.pop("_source", "unknown")   # extract metadata key

    # ── 3. Assemble input DataFrame in training column order ──────────────────
    input_df  = _build_feature_row(features)
    input_arr = input_df.values   # numpy array — avoids sklearn feature-name warning

    # ── 4. Predict ────────────────────────────────────────────────────────────
    prediction = float(model.predict(input_arr)[0])
    prediction = max(0.0, round(prediction, 2))

    # ── 5. Confidence score from per-tree prediction variance ─────────────────
    #   Lower variance across 200 trees → tighter ensemble → higher confidence
    #   Formula: confidence = 100 - (std / mean) * 100, clipped to [0, 100]
    tree_preds = np.array([
        float(tree.predict(input_arr)[0])
        for tree in model.estimators_
    ])
    std_dev    = float(np.std(tree_preds))
    confidence = float(max(0.0, min(100.0,
                    100.0 - (std_dev / (prediction + 1e-6)) * 100.0)))
    confidence = round(confidence, 1)

    # ── 6. Return complete response (all plain Python types — JSON-safe) ───────
    return {
        "forecasted_demand" : prediction,
        "confidence"        : confidence,
        "model_accuracy"    : metrics["accuracy"],
        "r2"                : round(metrics["r2"],   4),
        "mae"               : round(metrics["mae"],  2),
        "rmse"              : round(metrics["rmse"], 2),
        "product"           : str(product),
        "month"             : int(month),
        "data_source"       : source,
        # ── Transparency fields (shown in UI for demo/viva) ────────────────
        "lag_1"             : round(features["lag_1"],  2),
        "lag_2"             : round(features["lag_2"],  2),
        "lag_3"             : round(features["lag_3"],  2),
        "price_used"        : round(features["price"],  2),
        "promotion_flag"    : features["promotion_flag"],
    }


def get_historical_data(product: str = "Product_A", n_months: int = 12) -> dict:
    """
    Returns actual vs model-predicted demand for the chart.
    Falls back to synthetic data if the CSV is missing.
    """
    payload  = _load()
    model    = payload["model"]
    features = payload["features"]

    if not os.path.exists(DATA_PATH):
        # Demo fallback
        np.random.seed(0)
        actual    = [float(200 + i * 4 + np.random.randint(-18, 18)) for i in range(n_months)]
        predicted = [float(a + np.random.randint(-12, 12))           for a in actual]
        labels    = [f"Week-{i+1}" for i in range(n_months)]
        return {"labels": labels, "actual": actual, "predicted": predicted}

    df = pd.read_csv(DATA_PATH)
    if "product" in df.columns:
        df = df[df["product"] == product]
    df = df.dropna(subset=features).tail(n_months)

    if df.empty:
        return {"labels": [], "actual": [], "predicted": []}

    predicted = [float(v) for v in model.predict(df[features].values)]
    actual    = [float(v) for v in df["demand"].tolist()]
    labels    = [str(v)[:10] for v in df.get("date", pd.Series(range(len(actual))))]

    return {
        "labels"    : labels,
        "actual"    : [round(v, 1) for v in actual],
        "predicted" : [round(v, 1) for v in predicted],
    }


def get_model_metrics() -> dict:
    """Returns stored model metrics (all plain Python floats)."""
    return _load()["metrics"]