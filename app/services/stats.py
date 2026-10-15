"""Aggregates for the dashboard. Plain SQL + pandas over the mock dataset."""

import sqlite3
from datetime import date, timedelta

import pandas as pd

from app.features import load_booked, load_history
from app.services.predictor import NoShowPredictor

LEAD_BUCKETS = [(-1, 1, "0–1 days"), (1, 7, "2–7 days"), (7, 14, "8–14 days"), (14, 30, "15–30 days"),
                (30, 10_000, "30+ days")]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def _rate(no_shows: int, total: int) -> float:
    return round(no_shows / total, 4) if total else 0.0


def dashboard_stats(conn: sqlite3.Connection, predictor: NoShowPredictor, today: date | None = None,
                    upcoming_days: int = 7) -> dict:
    today = today or date.today()
    hist = load_history(conn)
    total, no_shows = len(hist), int(hist["no_show"].sum())

    start = pd.to_datetime(hist["start_time"])
    monthly = (hist.assign(month=start.dt.strftime("%Y-%m"))
               .groupby("month")["no_show"].agg(n="count", ns="sum").reset_index())
    weekday = (hist.assign(day=start.dt.weekday).groupby("day")["no_show"].agg(n="count", ns="sum").reset_index())
    dept = hist.groupby("department")["no_show"].agg(n="count", ns="sum").reset_index()
    lead_labels = pd.cut(hist["lead_time_days"], bins=[b[0] for b in LEAD_BUCKETS] + [LEAD_BUCKETS[-1][1]],
                         labels=[b[2] for b in LEAD_BUCKETS])
    lead = hist.assign(bucket=lead_labels).groupby("bucket", observed=False)["no_show"].agg(n="count", ns="sum").reset_index()
    sms = hist.groupby("sms_reminder")["no_show"].agg(n="count", ns="sum").reset_index()

    # Upcoming week: bookings per day and how many the model flags as high risk.
    upcoming = []
    for offset in range(upcoming_days):
        day = today + timedelta(days=offset)
        df = load_booked(conn, day=day.isoformat())
        preds = predictor.predict(df) if not df.empty else []
        upcoming.append({
            "date": day.isoformat(),
            "booked": len(preds),
            "high_risk": sum(p.risk == "high" for p in preds),
            "expected_no_shows": round(sum(p.probability for p in preds), 1),
        })

    chat = conn.execute(
        "SELECT COUNT(*) AS total, COALESCE(SUM(cached), 0) AS cached,"
        " COALESCE(SUM(source = 'llm'), 0) AS llm, COALESCE(SUM(source = 'fallback'), 0) AS fallback,"
        " COALESCE(SUM(error IS NOT NULL), 0) AS errors, AVG(latency_ms) AS avg_latency FROM chat_logs"
    ).fetchone()

    return {
        "totals": {
            "appointments": total,
            "completed": total - no_shows,
            "no_shows": no_shows,
            "no_show_rate": _rate(no_shows, total),
            "period": [hist["start_time"].min()[:10], hist["start_time"].max()[:10]],
            "upcoming_booked": sum(d["booked"] for d in upcoming),
            "upcoming_high_risk": sum(d["high_risk"] for d in upcoming),
        },
        "monthly": [{"month": r.month, "appointments": int(r.n), "no_shows": int(r.ns),
                     "no_show_rate": _rate(r.ns, r.n)} for r in monthly.itertuples()],
        "by_weekday": [{"day": WEEKDAYS[int(r.day)], "appointments": int(r.n), "no_shows": int(r.ns),
                        "no_show_rate": _rate(r.ns, r.n)} for r in weekday.itertuples()],
        "by_department": sorted(
            [{"department": r.department, "appointments": int(r.n), "no_shows": int(r.ns),
              "no_show_rate": _rate(r.ns, r.n)} for r in dept.itertuples()],
            key=lambda d: d["no_show_rate"], reverse=True),
        "by_lead_time": [{"bucket": str(r.bucket), "appointments": int(r.n), "no_shows": int(r.ns),
                          "no_show_rate": _rate(r.ns, r.n)} for r in lead.itertuples()],
        "by_sms_reminder": [{"sms_reminder": bool(r.sms_reminder), "appointments": int(r.n),
                             "no_shows": int(r.ns), "no_show_rate": _rate(r.ns, r.n)} for r in sms.itertuples()],
        "upcoming": upcoming,
        "chat": {
            "queries": chat["total"], "cache_hits": chat["cached"], "llm": chat["llm"],
            "fallback": chat["fallback"], "errors": chat["errors"],
            "avg_latency_ms": round(chat["avg_latency"], 1) if chat["avg_latency"] is not None else None,
        },
        "model": {
            "name": predictor.name,
            "roc_auc": predictor.metrics["results"][predictor.name]["roc_auc"],
            "brier": predictor.metrics["results"][predictor.name]["brier"],
            "baseline_brier": predictor.metrics["results"]["baseline_constant"]["brier"],
        },
    }
