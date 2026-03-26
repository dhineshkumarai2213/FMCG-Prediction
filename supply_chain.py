"""
supply_chain.py — Supply Chain Dispatch Optimization
=====================================================
Optimizes warehouse → distributor → retailer dispatch.
Uses a greedy priority-based dispatch algorithm:
  Priority = (demand_urgency × 0.5) + (distance_penalty × 0.3) + (stock_risk × 0.2)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DispatchNode:
    """Represents a retailer/distribution center needing supply."""
    name           : str
    location       : str
    current_stock  : float
    weekly_demand  : float
    distance_km    : float
    priority_score : float = field(init=False)

    def __post_init__(self):
        # Stock coverage in days
        days_cover      = (self.current_stock / self.weekly_demand * 7) if self.weekly_demand > 0 else 30
        urgency         = max(0, (14 - days_cover) / 14)        # 0–1: higher if <14 days stock
        distance_pen    = min(1, self.distance_km / 500)         # 0–1: farther = higher penalty
        stock_risk      = 1 - min(1, self.current_stock / (self.weekly_demand * 2))

        self.priority_score = round(
            urgency * 0.5 + distance_pen * 0.3 + stock_risk * 0.2, 4
        )


def optimize_dispatch(warehouse_stock: float, nodes: List[dict]) -> dict:
    """
    Greedy dispatch: sort nodes by priority, allocate stock until exhausted.

    Parameters
    ----------
    warehouse_stock : Total units available for dispatch
    nodes           : List of retailer dicts (name, location, current_stock, weekly_demand, distance_km)

    Returns
    -------
    dict with dispatch plan, efficiency metrics
    """
    dispatch_nodes = [DispatchNode(**n) for n in nodes]
    dispatch_nodes.sort(key=lambda x: x.priority_score, reverse=True)

    remaining_stock = warehouse_stock
    plan            = []
    total_dispatched = 0
    fulfilled_count  = 0

    for node in dispatch_nodes:
        # Ideal dispatch = 2 weeks of demand to cover lead time
        ideal_qty  = node.weekly_demand * 2
        dispatched = min(ideal_qty, remaining_stock)
        dispatched = round(dispatched, 2)

        status = "✅ Fulfilled" if dispatched >= ideal_qty else (
                 "⚠️ Partial"  if dispatched > 0 else "❌ Skipped")
        if dispatched >= ideal_qty:
            fulfilled_count += 1

        plan.append({
            "node"          : node.name,
            "location"      : node.location,
            "priority_score": node.priority_score,
            "ideal_qty"     : round(ideal_qty, 2),
            "dispatched_qty": dispatched,
            "status"        : status,
            "days_after"    : round((node.current_stock + dispatched) / node.weekly_demand * 7, 1)
                              if node.weekly_demand > 0 else 999,
        })

        remaining_stock  -= dispatched
        total_dispatched += dispatched
        if remaining_stock <= 0:
            remaining_stock = 0
            break

    efficiency = round((fulfilled_count / len(dispatch_nodes)) * 100, 1) if dispatch_nodes else 0

    return {
        "dispatch_plan"    : plan,
        "warehouse_stock"  : warehouse_stock,
        "total_dispatched" : round(total_dispatched, 2),
        "remaining_stock"  : round(remaining_stock, 2),
        "nodes_fulfilled"  : fulfilled_count,
        "total_nodes"      : len(dispatch_nodes),
        "efficiency_pct"   : efficiency,
    }


def get_sample_dispatch() -> dict:
    """Returns a sample dispatch scenario for demo purposes."""
    nodes = [
        {"name": "Retailer Mumbai Central", "location": "Mumbai",   "current_stock": 120, "weekly_demand": 80,  "distance_km": 50},
        {"name": "Retailer Pune Hub",        "location": "Pune",     "current_stock": 40,  "weekly_demand": 60,  "distance_km": 150},
        {"name": "Retailer Nashik Depot",    "location": "Nashik",   "current_stock": 200, "weekly_demand": 50,  "distance_km": 220},
        {"name": "Retailer Surat Branch",    "location": "Surat",    "current_stock": 15,  "weekly_demand": 90,  "distance_km": 280},
        {"name": "Retailer Ahmedabad Dist.", "location": "Ahmedabad","current_stock": 80,  "weekly_demand": 70,  "distance_km": 530},
    ]
    return optimize_dispatch(warehouse_stock=800, nodes=nodes)