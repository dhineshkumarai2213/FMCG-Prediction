"""
anomaly_detection.py — Risk & Anomaly Detection Module
=======================================================
Uses two complementary methods:
  1. Z-Score   → flags values > 2.5 standard deviations from mean
  2. IQR Fence → flags values outside Q1 - 1.5×IQR … Q3 + 1.5×IQR
Both methods together reduce false positives.
"""

import numpy as np
from typing import List


def zscore_anomalies(values: List[float], threshold: float = 2.5) -> List[bool]:
    """Returns boolean mask: True = anomaly."""
    arr  = np.array(values, dtype=float)
    mean = np.mean(arr)
    std  = np.std(arr)
    if std == 0:
        return [False] * len(values)
    z_scores = np.abs((arr - mean) / std)
    return (z_scores > threshold).tolist()


def iqr_anomalies(values: List[float], multiplier: float = 1.5) -> List[bool]:
    """Returns boolean mask using IQR fence method."""
    arr = np.array(values, dtype=float)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr     = q3 - q1
    lower   = q1 - multiplier * iqr
    upper   = q3 + multiplier * iqr
    return ((arr < lower) | (arr > upper)).tolist()


def detect_anomalies(dates: List[str], values: List[float]) -> dict:
    """
    Combines Z-score + IQR for robust anomaly detection.

    Returns
    -------
    dict with per-point flags, summary, and severity
    """
    zs_flags  = zscore_anomalies(values)
    iqr_flags = iqr_anomalies(values)

    # Anomaly if flagged by EITHER method (union for higher recall)
    combined = [z or i for z, i in zip(zs_flags, iqr_flags)]

    arr  = np.array(values)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))

    points = []
    alerts = []
    for idx, (date, val, flag) in enumerate(zip(dates, values, combined)):
        deviation = round(abs(val - mean) / (std + 1e-9), 2)
        severity  = "high" if deviation > 3 else ("medium" if deviation > 2 else "low")

        points.append({
            "date"     : date,
            "value"    : val,
            "anomaly"  : flag,
            "deviation": deviation,
            "severity" : severity if flag else "none",
        })

        if flag:
            direction = "spike" if val > mean else "drop"
            alerts.append({
                "date"    : date,
                "message" : f"Demand {direction} detected ({deviation}σ deviation)",
                "value"   : val,
                "severity": severity,
            })

    total_anomalies = sum(combined)
    risk_score      = round(min(100, (total_anomalies / len(values)) * 100 * 3), 1)

    return {
        "points"         : points,
        "alerts"         : alerts,
        "total_anomalies": total_anomalies,
        "risk_score"     : risk_score,
        "risk_level"     : "HIGH" if risk_score > 40 else ("MEDIUM" if risk_score > 20 else "LOW"),
        "mean"           : round(mean, 2),
        "std"            : round(std, 2),
    }


def get_sample_anomaly_data() -> dict:
    """Generates sample demand series with injected anomalies for demo."""
    np.random.seed(7)
    n = 24
    dates  = [f"2024-{(i % 12)+1:02d}-01" for i in range(n)]
    values = [200 + 10 * np.sin(i) + np.random.normal(0, 10) for i in range(n)]

    # Inject 3 anomalies
    values[4]  *= 2.5
    values[11] *= 0.2
    values[18] *= 2.8

    return detect_anomalies(dates, [round(v, 2) for v in values])