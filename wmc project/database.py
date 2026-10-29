"""
Database Management Module
Wireless Network Congestion Prediction System
Provides persistent SQLite database operations for cell stations, telemetry records, and alerts.
"""

import os
import sqlite3
import datetime
from typing import Dict, List, Optional, Tuple, Any

DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "wireless_network.db")


def get_db_connection() -> sqlite3.Connection:
    """Creates a thread-safe connection to the SQLite database with dictionary cursor."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Initializes the database schema with relational tables and indexes."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table 1: Base Stations / Radio Cells
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS base_stations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cell_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        location TEXT NOT NULL,
        frequency_band TEXT NOT NULL,
        technology TEXT NOT NULL, -- 4G LTE, 5G NR, Wi-Fi 6
        status TEXT NOT NULL DEFAULT 'Active', -- Active, Maintenance, Degraded
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Table 2: Telemetry Measurements and Model Predictions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS telemetry_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cell_id TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        network_load_pct REAL NOT NULL,
        throughput_mbps REAL NOT NULL,
        latency_ms REAL NOT NULL,
        packet_loss_pct REAL NOT NULL,
        predicted_congestion TEXT NOT NULL, -- Low, Medium, High
        confidence_pct REAL NOT NULL,
        model_used TEXT NOT NULL,
        prob_low REAL NOT NULL DEFAULT 0.0,
        prob_medium REAL NOT NULL DEFAULT 0.0,
        prob_high REAL NOT NULL DEFAULT 0.0,
        is_simulation INTEGER NOT NULL DEFAULT 0,
        FOREIGN KEY (cell_id) REFERENCES base_stations(cell_id) ON DELETE CASCADE
    );
    """)

    # Table 3: Congestion Alerts
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        telemetry_id INTEGER,
        cell_id TEXT NOT NULL,
        severity TEXT NOT NULL, -- Warning, Critical
        message TEXT NOT NULL,
        is_acknowledged INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (telemetry_id) REFERENCES telemetry_logs(id) ON DELETE CASCADE
    );
    """)

    # Create Indexes for fast querying on timestamps and congestion classes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_logs(timestamp);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_cell ON telemetry_logs(cell_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_congestion ON telemetry_logs(predicted_congestion);")

    conn.commit()

    # Seed initial base stations if none exist
    cursor.execute("SELECT COUNT(*) FROM base_stations;")
    if cursor.fetchone()[0] == 0:
        seed_base_stations(cursor)
        conn.commit()

    conn.close()
    print(f"[+] SQLite Database initialized successfully at: {DB_PATH}")


def seed_base_stations(cursor):
    """Seeds default realistic wireless base stations."""
    initial_stations = [
        ("BS-101-North", "North Commercial Plaza", "Downtown Commercial Core", "3.5 GHz (n78)", "5G NR", "Active"),
        ("BS-102-Central", "Central Metro Station", "Main Transit Hub", "2.6 GHz (Band 7)", "4G LTE", "Active"),
        ("BS-103-Campus", "Tech University Sector A", "Academic Campus Zone", "5.0 GHz (UNII)", "Wi-Fi 6", "Active"),
        ("BS-104-South", "South Industrial Park", "Logistics & Manufacturing Zone", "700 MHz (Band 28)", "4G LTE", "Active"),
        ("BS-105-Stadium", "City Sports Arena", "High-Density Event Venue", "28 GHz (mmWave)", "5G NR", "Active")
    ]
    cursor.executemany("""
    INSERT INTO base_stations (cell_id, name, location, frequency_band, technology, status)
    VALUES (?, ?, ?, ?, ?, ?);
    """, initial_stations)


# ==============================================================================
# Base Station Operations
# ==============================================================================

def get_all_base_stations() -> List[Dict[str, Any]]:
    """Retrieves all registered base stations."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM base_stations ORDER BY cell_id ASC;")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def add_base_station(cell_id: str, name: str, location: str, frequency_band: str, technology: str) -> Tuple[bool, str]:
    """Adds a new base station node to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO base_stations (cell_id, name, location, frequency_band, technology, status)
        VALUES (?, ?, ?, ?, ?, 'Active');
        """, (cell_id.strip(), name.strip(), location.strip(), frequency_band.strip(), technology.strip()))
        conn.commit()
        return True, "Base station added successfully."
    except sqlite3.IntegrityError:
        return False, f"Base station with ID '{cell_id}' already exists."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


# ==============================================================================
# Telemetry Log Operations
# ==============================================================================

def save_telemetry_record(
    cell_id: str,
    network_load_pct: float,
    throughput_mbps: float,
    latency_ms: float,
    packet_loss_pct: float,
    predicted_congestion: str,
    confidence_pct: float,
    model_used: str,
    probs: Dict[str, float],
    is_simulation: bool = False
) -> int:
    """
    Saves a telemetry measurement and its ML prediction to the database.
    Automatically generates an alert if congestion is Medium or High.
    Returns the newly inserted record ID.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Ensure cell_id exists in base_stations (or insert an ad-hoc one if unknown)
    cursor.execute("SELECT id FROM base_stations WHERE cell_id = ?;", (cell_id,))
    if not cursor.fetchone():
        cursor.execute("""
        INSERT OR IGNORE INTO base_stations (cell_id, name, location, frequency_band, technology, status)
        VALUES (?, ?, 'Dynamic Node', 'Dynamic Band', 'Wireless RAN', 'Active');
        """, (cell_id, f"Node {cell_id}"))

    cursor.execute("""
    INSERT INTO telemetry_logs (
        cell_id, network_load_pct, throughput_mbps, latency_ms, packet_loss_pct,
        predicted_congestion, confidence_pct, model_used,
        prob_low, prob_medium, prob_high, is_simulation
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        cell_id,
        network_load_pct,
        throughput_mbps,
        latency_ms,
        packet_loss_pct,
        predicted_congestion,
        confidence_pct,
        model_used,
        probs.get("Low", 0.0),
        probs.get("Medium", 0.0),
        probs.get("High", 0.0),
        1 if is_simulation else 0
    ))

    telemetry_id = cursor.lastrowid

    # Automatically create alert for Medium or High congestion
    if predicted_congestion == "High":
        cursor.execute("""
        INSERT INTO alerts (telemetry_id, cell_id, severity, message)
        VALUES (?, ?, 'Critical', ?);
        """, (
            telemetry_id,
            cell_id,
            f"Critical Congestion on {cell_id}: Load {network_load_pct}%, Latency {latency_ms}ms, Loss {packet_loss_pct}%"
        ))
    elif predicted_congestion == "Medium":
        cursor.execute("""
        INSERT INTO alerts (telemetry_id, cell_id, severity, message)
        VALUES (?, ?, 'Warning', ?);
        """, (
            telemetry_id,
            cell_id,
            f"Moderate Congestion warning on {cell_id}: Load {network_load_pct}%, Latency {latency_ms}ms"
        ))

    conn.commit()
    conn.close()
    return telemetry_id


def get_recent_telemetry(limit: int = 25, cell_id: Optional[str] = None, congestion: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves recent telemetry records with optional filtering."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
    SELECT t.*, b.name as cell_name, b.technology 
    FROM telemetry_logs t
    LEFT JOIN base_stations b ON t.cell_id = b.cell_id
    WHERE 1=1
    """
    params = []

    if cell_id:
        query += " AND t.cell_id = ?"
        params.append(cell_id)

    if congestion:
        query += " AND t.predicted_congestion = ?"
        params.append(congestion)

    query += " ORDER BY t.timestamp DESC, t.id DESC LIMIT ?;"
    params.append(limit)

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_telemetry_statistics() -> Dict[str, Any]:
    """Computes real-time dynamic statistics across all logged telemetry in SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Total counts and averages
    cursor.execute("""
    SELECT 
        COUNT(*) as total_logs,
        COALESCE(AVG(network_load_pct), 0.0) as avg_load,
        COALESCE(AVG(throughput_mbps), 0.0) as avg_throughput,
        COALESCE(AVG(latency_ms), 0.0) as avg_latency,
        COALESCE(AVG(packet_loss_pct), 0.0) as avg_packet_loss
    FROM telemetry_logs;
    """)
    stats_row = dict(cursor.fetchone())

    # Counts by congestion class
    cursor.execute("""
    SELECT predicted_congestion, COUNT(*) as count
    FROM telemetry_logs
    GROUP BY predicted_congestion;
    """)
    class_counts = {"Low": 0, "Medium": 0, "High": 0}
    for row in cursor.fetchall():
        class_counts[row["predicted_congestion"]] = row["count"]

    # Active alert count
    cursor.execute("SELECT COUNT(*) FROM alerts WHERE is_acknowledged = 0;")
    active_alerts = cursor.fetchone()[0]

    # Total registered stations
    cursor.execute("SELECT COUNT(*) FROM base_stations;")
    total_stations = cursor.fetchone()[0]

    conn.close()

    return {
        "total_records": stats_row["total_logs"],
        "class_counts": class_counts,
        "averages": {
            "load_pct": round(stats_row["avg_load"], 2),
            "throughput_mbps": round(stats_row["avg_throughput"], 2),
            "latency_ms": round(stats_row["avg_latency"], 2),
            "packet_loss_pct": round(stats_row["avg_packet_loss"], 2)
        },
        "active_alerts_count": active_alerts,
        "total_stations": total_stations
    }


def get_timeseries_data(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves chronological telemetry measurements for live dynamic charts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT 
        id,
        cell_id,
        strftime('%H:%M:%S', timestamp) as time_label,
        network_load_pct,
        throughput_mbps,
        latency_ms,
        packet_loss_pct,
        predicted_congestion
    FROM telemetry_logs
    ORDER BY timestamp DESC, id DESC
    LIMIT ?;
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    # Return chronologically ascending for charts (left-to-right)
    return list(reversed(rows))


def delete_telemetry_record(record_id: int) -> bool:
    """Deletes a specific telemetry record from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM telemetry_logs WHERE id = ?;", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def clear_all_telemetry() -> int:
    """Clears all telemetry records and associated alerts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM alerts;")
    cursor.execute("DELETE FROM telemetry_logs;")
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted_count


# ==============================================================================
# Alert Operations
# ==============================================================================

def get_active_alerts(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieves unacknowledged congestion alerts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM alerts 
    WHERE is_acknowledged = 0 
    ORDER BY created_at DESC, id DESC 
    LIMIT ?;
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def acknowledge_alert(alert_id: int) -> bool:
    """Marks an alert as acknowledged in the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE alerts SET is_acknowledged = 1 WHERE id = ?;", (alert_id,))
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


# Initialize database when module is imported
init_db()
