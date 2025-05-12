# db_utils.py

import os
import sqlite3
import json
import streamlit as st
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "users.db")

@st.cache_resource
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username      TEXT PRIMARY KEY,
            salt          BLOB NOT NULL,
            password_hash BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS projects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT,
            name        TEXT,
            client      TEXT,
            study       TEXT,
            phase       TEXT,
            description TEXT,
            type        TEXT,
            created_at  TEXT
        );
        CREATE TABLE IF NOT EXISTS project_state (
            project_id   INTEGER,
            key          TEXT,
            json_value   TEXT,
            updated_at   TEXT,
            PRIMARY KEY(project_id, key)
        );
    """)
    conn.commit()
    return conn

def load_state(pid: int, key: str, default=None):
    row = get_db().execute(
        "SELECT json_value FROM project_state WHERE project_id=? AND key=?",
        (pid, key)
    ).fetchone()
    return json.loads(row[0]) if row else default

def save_state(pid: int, key: str, obj):
    payload = json.dumps(obj, default=str)
    db = get_db()
    db.execute("""
        INSERT INTO project_state(project_id, key, json_value, updated_at)
        VALUES (?,?,?,?)
        ON CONFLICT(project_id, key) DO UPDATE SET
          json_value = excluded.json_value,
          updated_at = excluded.updated_at
    """, (pid, key, payload, datetime.now().isoformat(timespec="seconds")))
    db.commit()
