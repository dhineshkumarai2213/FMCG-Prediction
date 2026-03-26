"""
app.py  —  FMCG AI · Flask REST API Server  (v2 — Fully Automated Forecast)
=============================================================================
Start:
    cd backend && python app.py

WHAT CHANGED IN v2
──────────────────
/api/forecast now accepts ONLY:
    { "product": "Product_A", "month": 6 }

All lag features, rolling means, price, and promotion_flag are generated
automatically by forecast_model.py — no user input required for these.

ENDPOINTS
─────────
GET  /                      → login page
GET  /dashboard             → main dashboard (session-required)
POST /api/login             → authenticate
POST /api/logout            → clear session
POST /api/forecast          → automated demand prediction ← SIMPLIFIED v2
POST /api/inventory         → inventory optimisation
GET  /api/supply-chain      → dispatch plan
GET  /api/anomalies         → anomaly + risk detection
GET  /api/history           → actual vs predicted chart data
GET  /api/metrics           → model performance metrics
GET  /api/recent-forecasts  → last 10 logged forecasts
"""

import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for

try:
    from flask_cors import CORS
    _CORS_AVAILABLE = True
except ImportError:
    _CORS_AVAILABLE = False

from database          import init_db, verify_user, log_forecast, log_inventory, get_recent_forecasts
from forecast_model    import predict_demand, get_historical_data, get_model_metrics
from inventory_logic   import full_inventory_analysis
from supply_chain      import get_sample_dispatch
from anomaly_detection import get_sample_anomaly_data

# ── App initialisation ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder = os.path.join(BASE_DIR, "..", "frontend", "templates"),
    static_folder   = os.path.join(BASE_DIR, "..", "frontend", "static"),
)
app.secret_key = "fmcg_ai_secret_key_change_before_production_2024"

if _CORS_AVAILABLE:
    CORS(app, supports_credentials=True)

init_db()   # create tables + seed users on every cold start (idempotent)


# ─────────────────────────────────────────────────────────────────────────────
# Helper — safe JSON body extraction
#
# Handles three bad-input cases so no endpoint ever sees them:
#   1. Body is not JSON at all         → 400 with clear message
#   2. Body is a JSON array [{ … }]    → extract [0] silently (old frontend bug)
#   3. Body is not dict or list        → 400
# ─────────────────────────────────────────────────────────────────────────────
def _safe_get_json():
    """
    Returns (data_dict, None) on success.
    Returns (None, flask_response) on any parse / type error.
    """
    raw = request.get_json(silent=True, force=True)   # force=True: parse even without Content-Type

    if raw is None:
        return None, (jsonify({"error": "Request body must be valid JSON."}), 400)

    # Normalise list → dict (handles legacy frontend that wraps in [ ])
    if isinstance(raw, list):
        if not raw:
            return None, (jsonify({"error": "Payload array is empty."}), 400)
        raw = raw[0]

    if not isinstance(raw, dict):
        return None, (jsonify({
            "error": f"Expected a JSON object, received {type(raw).__name__}."
        }), 400)

    return raw, None


# ─────────────────────────────────────────────────────────────────────────────
# Page routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("index"))
    return render_template("dashboard.html", user=session["user"])


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/login", methods=["POST"])
def api_login():
    data, err = _safe_get_json()
    if err:
        return err

    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required."}), 400

    user = verify_user(username, password)
    if user:
        session["user"] = {"username": user["username"], "role": user["role"]}
        return jsonify({"success": True, "role": user["role"], "redirect": "/dashboard"})

    return jsonify({"success": False, "message": "Invalid username or password."}), 401


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})


# ─────────────────────────────────────────────────────────────────────────────
# /api/forecast  —  v2: FULLY AUTOMATED
#
# REQUIRED input  (from frontend):
#     { "product": "Product_A", "month": 6 }
#
# REMOVED inputs  (handled automatically by forecast_model.py):
#     lag_1, lag_2, lag_3, price, promotion_flag,
#     rolling_mean_3, rolling_mean_6, quarter, week_of_year
#
# RESPONSE:
#     {
#         "forecasted_demand": 247.3,
#         "confidence"       : 91.2,
#         "model_accuracy"   : 81.07,
#         "r2"               : 0.8107,
#         "mae"              : 15.49,
#         "rmse"             : 19.59,
#         "product"          : "Product_A",
#         "month"            : 6,
#         "data_source"      : "csv",       ← "csv" or "synthetic"
#         "lag_1"            : 235.1,       ← auto-generated (shown in UI)
#         "lag_2"            : 224.9,
#         "lag_3"            : 244.5,
#         "price_used"       : 52.0,
#         "promotion_flag"   : 0
#     }
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/forecast", methods=["POST"])
def api_forecast():
    if "user" not in session:
        return jsonify({"error": "Unauthorised. Please log in."}), 401

    # ── Parse and validate request body ──────────────────────────────────────
    data, err = _safe_get_json()
    if err:
        return err

    # ── Extract the only two required user inputs ─────────────────────────────
    product = str(data.get("product", "Product_A")).strip()
    try:
        month = int(data.get("month", 6))
    except (ValueError, TypeError):
        return jsonify({"error": "month must be an integer between 1 and 12."}), 400

    if not (1 <= month <= 12):
        return jsonify({"error": f"month must be 1–12, got {month}."}), 400

    valid_products = ["Product_A", "Product_B", "Product_C", "Product_D"]
    if product not in valid_products:
        return jsonify({
            "error"  : f"Unknown product '{product}'.",
            "valid"  : valid_products,
        }), 400

    # ── Run automated forecast ────────────────────────────────────────────────
    try:
        result = predict_demand(product=product, month=month)

    except FileNotFoundError as exc:
        return jsonify({
            "error" : str(exc),
            "action": "Run  python train_model.py  in the backend folder."
        }), 503

    except Exception as exc:
        return jsonify({"error": f"Forecast failed: {exc}"}), 500

    # ── Log to database (non-blocking) ────────────────────────────────────────
    try:
        log_forecast(
            product    = result["product"],
            month      = result["month"],
            price      = result["price_used"],
            promotion  = result["promotion_flag"],
            forecast   = result["forecasted_demand"],
            confidence = result["confidence"],
        )
    except Exception:
        pass   # log failure never breaks the API

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
# /api/inventory
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/inventory", methods=["POST"])
def api_inventory():
    if "user" not in session:
        return jsonify({"error": "Unauthorised."}), 401

    data, err = _safe_get_json()
    if err:
        return err

    try:
        result = full_inventory_analysis(
            current_stock         = float(data.get("current_stock",          500)),
            forecasted_demand     = float(data.get("forecasted_demand",      200)),
            lead_time_days        = int  (data.get("lead_time_days",           7)),
            ordering_cost         = float(data.get("ordering_cost",          500)),
            holding_cost_per_unit = float(data.get("holding_cost_per_unit",  2.0)),
        )
    except Exception as exc:
        return jsonify({"error": f"Inventory calculation failed: {exc}"}), 500

    try:
        log_inventory(
            product       = str(data.get("product", "Unknown")),
            current_stock = float(data.get("current_stock", 0)),
            reorder_point = result["reorder_point"],
            reorder_qty   = result["reorder_quantity"],
            alert_level   = result["alert_level"],
        )
    except Exception:
        pass

    return jsonify(result)


# ─────────────────────────────────────────────────────────────────────────────
# /api/supply-chain
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/supply-chain", methods=["GET"])
def api_supply_chain():
    if "user" not in session:
        return jsonify({"error": "Unauthorised."}), 401
    try:
        return jsonify(get_sample_dispatch())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# /api/anomalies
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/anomalies", methods=["GET"])
def api_anomalies():
    if "user" not in session:
        return jsonify({"error": "Unauthorised."}), 401
    try:
        return jsonify(get_sample_anomaly_data())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# /api/history
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
def api_history():
    if "user" not in session:
        return jsonify({"error": "Unauthorised."}), 401
    product = request.args.get("product", "Product_A")
    n       = int(request.args.get("n", 12))
    try:
        return jsonify(get_historical_data(product, n))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# /api/metrics
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    if "user" not in session:
        return jsonify({"error": "Unauthorised."}), 401
    try:
        return jsonify(get_model_metrics())
    except FileNotFoundError as exc:
        return jsonify({
            "error" : str(exc),
            "action": "Run  python train_model.py  first."
        }), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# /api/recent-forecasts
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/recent-forecasts", methods=["GET"])
def api_recent_forecasts():
    if "user" not in session:
        return jsonify({"error": "Unauthorised."}), 401
    return jsonify(get_recent_forecasts())


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  FMCG AI v2  →  http://127.0.0.1:5000")
    print("  admin / admin123   |   analyst / analyst123")
    print("=" * 55)
    app.run(debug=True, port=5000, use_reloader=True)