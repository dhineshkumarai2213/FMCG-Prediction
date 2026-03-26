"""
train_model.py  ─  FMCG AI · Demand Forecasting Model Trainer
===============================================================
Run ONCE before starting the Flask server:
    cd backend
    python train_model.py

What this script does
─────────────────────
1. Generates a synthetic FMCG weekly sales dataset (4 products × 500 weeks)
2. Engineers 10 features used by the model
3. Trains a RandomForestRegressor with a per-product temporal split
4. Computes MAE, RMSE, R² — all stored as plain Python floats (NOT np.float64)
5. Saves ONE dictionary to demand_model.pkl:
       {
           "model"   : <RandomForestRegressor>,
           "features": [list of feature column names],
           "metrics" : {"mae": float, "rmse": float, "r2": float, "accuracy": float}
       }
6. Saves the processed dataset to data/sales_data.csv

WHY THE DICT MATTERS
─────────────────────
The .pkl was previously saved as a bare model object in some versions,
which caused joblib.load() to return a RandomForestRegressor directly.
When forecast_model.py then did  payload["model"]  on a non-dict object,
Python raised:
    TypeError: list indices must be integers or slices, not str
Wrapping everything in a single dict with fixed keys eliminates this.

VIVA ONE-LINER
──────────────
"We wrap the model, feature list, and metrics in a single dictionary
before pickling so that every downstream module can unpack them safely
using named keys instead of positional indexing."
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics   import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Resolve paths relative to THIS file — works regardless of cwd
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "model")
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(MODEL_DIR, "demand_model.pkl")
DATA_PATH  = os.path.join(DATA_DIR,  "sales_data.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Generate synthetic FMCG weekly sales data
# ─────────────────────────────────────────────────────────────────────────────
print("Generating synthetic FMCG dataset ...")
np.random.seed(42)

N_WEEKS     = 500
PRODUCTS    = ["Product_A", "Product_B", "Product_C", "Product_D"]
CATEGORIES  = ["Beverages",  "Snacks",    "Dairy",     "Personal_Care"]
BASE_DEMAND = [200,           250,          180,          220]

dates   = pd.date_range(start="2020-01-01", periods=N_WEEKS, freq="W")
records = []

for j, (prod, cat, base) in enumerate(zip(PRODUCTS, CATEGORIES, BASE_DEMAND)):
    for i, date in enumerate(dates):
        trend   = i * 0.08
        season  = base * 0.15 * np.sin(2 * np.pi * date.month / 12)
        promo   = int(np.random.random() < 0.18)
        price   = round(np.random.uniform(15, 90), 2)
        noise   = np.random.normal(0, base * 0.07)
        demand  = max(0, base + trend + season + promo * base * 0.25
                      - price * 0.25 + noise)
        records.append({
            "date"           : date,
            "product"        : prod,
            "category"       : cat,
            "demand"         : round(demand, 2),
            "price"          : price,
            "promotion_flag" : promo,
        })

df = pd.DataFrame(records).sort_values(["product", "date"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
print("Engineering features ...")

df["month"]        = df["date"].dt.month
df["quarter"]      = df["date"].dt.quarter
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

for lag in [1, 2, 3]:
    df[f"lag_{lag}"] = df.groupby("product")["demand"].shift(lag)

df["rolling_mean_3"] = (
    df.groupby("product")["demand"]
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)
df["rolling_mean_6"] = (
    df.groupby("product")["demand"]
    .transform(lambda x: x.shift(1).rolling(window=6, min_periods=1).mean())
)

df.dropna(subset=["lag_1", "lag_2", "lag_3"], inplace=True)
df.reset_index(drop=True, inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature list — ORDER IS FIXED, must match forecast_model.py exactly
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "month",           # 1-12: annual seasonality
    "quarter",         # 1-4:  broader seasonal grouping
    "week_of_year",    # 1-52: fine-grained position in year
    "lag_1",           # demand 1 week ago
    "lag_2",           # demand 2 weeks ago
    "lag_3",           # demand 3 weeks ago
    "rolling_mean_3",  # 3-week moving average (recent trend)
    "rolling_mean_6",  # 6-week moving average (medium trend)
    "price",           # unit price (demand elasticity)
    "promotion_flag",  # 0/1 — promotional week
]
TARGET = "demand"

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Per-product temporal train/test split  (80% train | 20% test)
# ─────────────────────────────────────────────────────────────────────────────
print("Splitting data (per-product temporal split, no shuffle) ...")
train_frames, test_frames = [], []
for prod in df["product"].unique():
    sub   = df[df["product"] == prod]
    split = int(len(sub) * 0.80)
    train_frames.append(sub.iloc[:split])
    test_frames .append(sub.iloc[split:])

train_df = pd.concat(train_frames).reset_index(drop=True)
test_df  = pd.concat(test_frames) .reset_index(drop=True)

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test,  y_test  = test_df [FEATURES], test_df [TARGET]

print(f"  Train rows: {len(X_train)} | Test rows: {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Train Random Forest Regressor
# ─────────────────────────────────────────────────────────────────────────────
print("Training RandomForestRegressor (200 trees) ...")
model = RandomForestRegressor(
    n_estimators    = 200,
    max_depth       = 12,
    min_samples_leaf= 5,
    max_features    = "sqrt",
    n_jobs          = -1,
    random_state    = 42,
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Evaluate — cast ALL metrics to plain Python float (never np.float64)
#
#     FIX EXPLANATION:
#     np.float64 is a numpy scalar, not a standard Python float.
#     Flask's jsonify uses json.dumps() which raises TypeError on np.float64
#     in some environments. float() converts any numeric type safely.
# ─────────────────────────────────────────────────────────────────────────────
y_pred   = model.predict(X_test)
mae      = float(mean_absolute_error(y_test, y_pred))
rmse     = float(math.sqrt(float(mean_squared_error(y_test, y_pred))))
r2       = float(r2_score(y_test, y_pred))
accuracy = float(round(r2 * 100, 2))

print(f"\n{'='*50}")
print(f"  MODEL PERFORMANCE (test set)")
print(f"{'='*50}")
print(f"  MAE      : {mae:.4f} units")
print(f"  RMSE     : {rmse:.4f} units")
print(f"  R2       : {r2:.4f}")
print(f"  Accuracy : {accuracy:.2f}%")
print(f"{'='*50}")

importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n  FEATURE IMPORTANCES:")
for feat, imp in importances.items():
    bar = "#" * int(imp * 40)
    print(f"  {feat:<20} {imp:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Save as a STRICT DICTIONARY — this is the only correct format
#
#     GUARANTEED STRUCTURE:
#     {
#         "model"   : RandomForestRegressor   <- the trained estimator
#         "features": list[str]               <- in training order
#         "metrics" : {                        <- all plain Python float
#             "mae"     : float,
#             "rmse"    : float,
#             "r2"      : float,
#             "accuracy": float,
#         }
#     }
# ─────────────────────────────────────────────────────────────────────────────
model_payload = {
    "model"   : model,
    "features": list(FEATURES),   # explicit list copy, no numpy array
    "metrics" : {
        "mae"     : mae,           # float, not np.float64
        "rmse"    : rmse,
        "r2"      : r2,
        "accuracy": accuracy,
    },
}

joblib.dump(model_payload, MODEL_PATH, compress=3)

# Immediate self-verification — crash here rather than silently at runtime
loaded = joblib.load(MODEL_PATH)
assert isinstance(loaded, dict),        "SAVE FAILED: pkl is not a dict"
assert set(loaded.keys()) == {"model", "features", "metrics"}, \
    "SAVE FAILED: unexpected keys in pkl"
assert loaded["features"] == FEATURES,  "SAVE FAILED: feature order mismatch"
assert all(isinstance(v, float) for v in loaded["metrics"].values()), \
    "SAVE FAILED: metrics contain non-float values"

print(f"\n[OK] Model verified and saved  -> {MODEL_PATH}")
print(f"[OK] Dataset saved             -> {DATA_PATH}")
df.to_csv(DATA_PATH, index=False)
print(f"\nNext step:  python app.py")