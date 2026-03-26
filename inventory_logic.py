"""
inventory_logic.py — Inventory Optimization Module
====================================================
Implements:
  - Economic Order Quantity (EOQ) — optimal order size to minimize cost
  - Reorder Point (ROP)           — when to place a new order
  - Safety Stock                  — buffer for demand uncertainty
  - Stockout alert logic
"""

import math


def calculate_safety_stock(avg_demand: float, std_demand: float,
                            lead_time_days: int, service_level: float = 0.95) -> float:
    """
    Safety Stock = Z × σ_demand × √lead_time
    Z: 1.65 for 95% service level, 2.05 for 98%, 2.33 for 99%
    """
    z_scores = {0.90: 1.28, 0.95: 1.65, 0.98: 2.05, 0.99: 2.33}
    z = z_scores.get(service_level, 1.65)
    return round(z * std_demand * math.sqrt(lead_time_days), 2)


def calculate_eoq(annual_demand: float, ordering_cost: float,
                  holding_cost_per_unit: float) -> float:
    """
    EOQ = √(2 × D × S / H)
    D = annual demand (units)
    S = ordering cost per order (₹)
    H = holding cost per unit per year (₹)
    """
    if holding_cost_per_unit <= 0:
        return 0.0
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
    return round(eoq, 2)


def calculate_reorder_point(avg_daily_demand: float, lead_time_days: int,
                             safety_stock: float) -> float:
    """
    ROP = (avg_daily_demand × lead_time) + safety_stock
    """
    return round(avg_daily_demand * lead_time_days + safety_stock, 2)


def get_inventory_status(current_stock: float, reorder_point: float,
                          eoq: float, forecasted_demand: float) -> dict:
    """
    Returns complete inventory analysis including alerts.
    """
    days_of_stock = round(current_stock / forecasted_demand, 1) if forecasted_demand > 0 else 999

    if current_stock <= 0:
        alert = "🔴 STOCKOUT"
        alert_level = "danger"
    elif current_stock < reorder_point * 0.5:
        alert = "🔴 CRITICAL — Order Immediately"
        alert_level = "danger"
    elif current_stock < reorder_point:
        alert = "🟡 REORDER NOW"
        alert_level = "warning"
    elif current_stock < reorder_point * 1.5:
        alert = "🟢 Monitor Stock"
        alert_level = "info"
    else:
        alert = "✅ Stock Optimal"
        alert_level = "success"

    return {
        "current_stock"    : current_stock,
        "reorder_point"    : reorder_point,
        "reorder_quantity" : eoq,
        "days_of_stock"    : days_of_stock,
        "alert"            : alert,
        "alert_level"      : alert_level,
        "should_reorder"   : current_stock < reorder_point,
    }


def full_inventory_analysis(current_stock: float, forecasted_demand: float,
                             lead_time_days: int = 7, ordering_cost: float = 500,
                             holding_cost_per_unit: float = 2.0,
                             std_demand: float = None) -> dict:
    """
    Master function — runs full inventory optimization pipeline.
    """
    if std_demand is None:
        std_demand = forecasted_demand * 0.15  # assume 15% variability

    annual_demand  = forecasted_demand * 52  # weekly → annual
    safety_stock   = calculate_safety_stock(forecasted_demand, std_demand, lead_time_days)
    eoq            = calculate_eoq(annual_demand, ordering_cost, holding_cost_per_unit)
    reorder_point  = calculate_reorder_point(forecasted_demand / 7, lead_time_days, safety_stock)
    status         = get_inventory_status(current_stock, reorder_point, eoq, forecasted_demand)

    return {
        **status,
        "safety_stock"  : safety_stock,
        "annual_demand" : round(annual_demand, 2),
        "lead_time_days": lead_time_days,
    }