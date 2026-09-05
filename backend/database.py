import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "audit.db")

def init_db():
    """Initializes the database and creates the audit_logs table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event JSON,
                candidates JSON,
                decision JSON,
                human_decision TEXT,
                apply_error TEXT
            )
        """)
        conn.commit()

def save_audit_log(log_entry: dict):
    """Saves a single audit log entry from the orchestrator into the database."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_logs (timestamp, event, candidates, decision, human_decision, apply_error)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            log_entry.get("timestamp"),
            json.dumps(log_entry.get("event")) if log_entry.get("event") else None,
            json.dumps(log_entry.get("candidates")) if log_entry.get("candidates") else None,
            json.dumps(log_entry.get("decision")) if log_entry.get("decision") else None,
            str(log_entry.get("human_decision")) if log_entry.get("human_decision") is not None else None,
            log_entry.get("apply_error")
        ))
        conn.commit()

def get_all_audit_logs():
    """Retrieves all audit logs, parsing the JSON fields back into dicts."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        
        logs = []
        for row in rows:
            log_dict = dict(row)
            # Parse JSON strings back to dicts
            for field in ["event", "candidates", "decision"]:
                if log_dict[field]:
                    log_dict[field] = json.loads(log_dict[field])
            logs.append(log_dict)
        return logs
