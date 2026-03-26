"""
database.py — SQLite Database Layer
=====================================
Creates tables, handles user auth (hashed passwords), and logs predictions.
"""

import os
import sqlite3
import hashlib
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "fmcg.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables and seed default admin user."""
    conn = get_connection()
    c    = conn.cursor()

    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT UNIQUE NOT NULL,
            password   TEXT NOT NULL,
            role       TEXT DEFAULT 'analyst',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Forecast log table
    c.execute("""
        CREATE TABLE IF NOT EXISTS forecast_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            product      TEXT,
            month        INTEGER,
            price        REAL,
            promotion    INTEGER,
            forecast     REAL,
            confidence   REAL,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Inventory log table
    c.execute("""
        CREATE TABLE IF NOT EXISTS inventory_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            product        TEXT,
            current_stock  REAL,
            reorder_point  REAL,
            reorder_qty    REAL,
            alert_level    TEXT,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Seed admin user (password: admin123)
    pwd_hash = _hash("admin123")
    c.execute("""
        INSERT OR IGNORE INTO users (username, password, role)
        VALUES (?, ?, ?)
    """, ("admin", pwd_hash, "admin"))

    # Seed analyst user (password: analyst123)
    c.execute("""
        INSERT OR IGNORE INTO users (username, password, role)
        VALUES (?, ?, ?)
    """, ("analyst", _hash("analyst123"), "analyst"))

    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_user(username: str, password: str):
    """Returns user row if credentials are valid, else None."""
    conn = get_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, _hash(password))
    ).fetchone()
    conn.close()
    return dict(user) if user else None


def log_forecast(product, month, price, promotion, forecast, confidence):
    conn = get_connection()
    conn.execute("""
        INSERT INTO forecast_log (product, month, price, promotion, forecast, confidence)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (product, month, price, promotion, forecast, confidence))
    conn.commit()
    conn.close()


def log_inventory(product, current_stock, reorder_point, reorder_qty, alert_level):
    conn = get_connection()
    conn.execute("""
        INSERT INTO inventory_log (product, current_stock, reorder_point, reorder_qty, alert_level)
        VALUES (?, ?, ?, ?, ?)
    """, (product, current_stock, reorder_point, reorder_qty, alert_level))
    conn.commit()
    conn.close()


def get_recent_forecasts(limit: int = 10):
    conn  = get_connection()
    rows  = conn.execute(
        "SELECT * FROM forecast_log ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]