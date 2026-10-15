"""SQLite access: connection helper and schema.

SQLite keeps the project zero-setup. Every query uses bound parameters
(never string formatting) so user input can't inject SQL.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS doctors (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    department    TEXT NOT NULL,
    working_days  TEXT NOT NULL          -- comma-separated weekday numbers, Mon=0
);

CREATE TABLE IF NOT EXISTS patients (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    age           INTEGER NOT NULL,
    distance_km   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS appointments (
    id            INTEGER PRIMARY KEY,
    patient_id    INTEGER NOT NULL REFERENCES patients(id),
    doctor_id     INTEGER NOT NULL REFERENCES doctors(id),
    start_time    TEXT NOT NULL,         -- ISO 8601, local clinic time
    booked_at     TEXT NOT NULL,
    sms_reminder  INTEGER NOT NULL,      -- 0/1
    status        TEXT NOT NULL CHECK (status IN ('booked', 'completed', 'no_show'))
);

CREATE INDEX IF NOT EXISTS idx_appt_doctor_time ON appointments (doctor_id, start_time);
CREATE INDEX IF NOT EXISTS idx_appt_patient ON appointments (patient_id, start_time);

CREATE TABLE IF NOT EXISTS chat_logs (
    id            INTEGER PRIMARY KEY,
    timestamp     TEXT NOT NULL,
    query         TEXT NOT NULL,
    intent        TEXT,
    source        TEXT,                  -- 'llm' or 'fallback'
    cached        INTEGER,
    latency_ms    REAL,
    error         TEXT
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or get_settings().database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


@contextmanager
def get_conn(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()
