"""Feature engineering shared by training and serving.

Keeping one definition avoids train/serve skew: the model sees exactly the
same columns, computed the same way, in both places.

"Prior" counts only use appointments that happened *before* the one being
scored, so the training data never leaks the future into the past.
"""

import sqlite3

import pandas as pd

NUMERIC_FEATURES = [
    "lead_time_days",
    "age",
    "distance_km",
    "prior_visits",
    "prior_no_shows",
    "prior_no_show_rate",
    "sms_reminder",
    "is_first_visit",
]
CATEGORICAL_FEATURES = ["weekday", "hour", "department"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "no_show"

# Window functions give each appointment the patient's history up to (not incl.) that row.
_BASE_SQL = """
WITH ordered AS (
    SELECT
        a.id                AS appointment_id,
        a.start_time,
        a.booked_at,
        a.status,
        a.sms_reminder,
        p.id                AS patient_id,
        p.name              AS patient_name,
        p.age,
        p.distance_km,
        d.name              AS doctor,
        d.department,
        COALESCE(SUM(a.status IN ('completed', 'no_show')) OVER w, 0) AS prior_visits,
        COALESCE(SUM(a.status = 'no_show') OVER w, 0)                 AS prior_no_shows
    FROM appointments a
    JOIN patients p ON p.id = a.patient_id
    JOIN doctors  d ON d.id = a.doctor_id
    WINDOW w AS (PARTITION BY a.patient_id ORDER BY a.start_time
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT * FROM ordered
"""


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw appointment rows into model features (pure, testable)."""
    out = df.copy()
    start = pd.to_datetime(out["start_time"])
    booked = pd.to_datetime(out["booked_at"])
    out["lead_time_days"] = ((start - booked).dt.total_seconds() / 86400).clip(lower=0).round(1)
    out["weekday"] = start.dt.weekday.astype(str)
    out["hour"] = start.dt.hour.astype(str)
    out["is_first_visit"] = (out["prior_visits"] == 0).astype(int)
    out["prior_no_show_rate"] = (out["prior_no_shows"] / out["prior_visits"].where(out["prior_visits"] > 0)).fillna(0.0)
    return out


def load_history(conn: sqlite3.Connection) -> pd.DataFrame:
    """Completed/no-show appointments with features and the label."""
    df = pd.read_sql_query(_BASE_SQL + " WHERE status IN ('completed', 'no_show') ORDER BY start_time", conn)
    df = add_derived_columns(df)
    df[TARGET] = (df["status"] == "no_show").astype(int)
    return df


def load_booked(conn: sqlite3.Connection, appointment_ids: list[int] | None = None,
                day: str | None = None) -> pd.DataFrame:
    """Upcoming ('booked') appointments with features, optionally filtered."""
    sql = _BASE_SQL + " WHERE status = 'booked'"
    params: list = []
    if appointment_ids is not None:
        if not appointment_ids:
            return pd.DataFrame()
        sql += f" AND appointment_id IN ({','.join('?' * len(appointment_ids))})"
        params.extend(appointment_ids)
    if day:
        sql += " AND start_time LIKE ?"
        params.append(f"{day}%")
    df = pd.read_sql_query(sql + " ORDER BY start_time", conn, params=params)
    return add_derived_columns(df) if not df.empty else df
