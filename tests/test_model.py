import json

import pandas as pd
import pytest

from app.features import FEATURES, add_derived_columns, load_booked, load_history
from app.services.predictor import ModelNotAvailableError, NoShowPredictor, risk_band
from tests.conftest import MONDAY

# --- features --------------------------------------------------------------


def test_prior_counts_only_use_earlier_appointments(conn):
    df = load_history(conn)
    patient = df[df["patient_id"] == df["patient_id"].value_counts().idxmax()].sort_values("start_time")
    assert patient["prior_visits"].tolist() == list(range(len(patient)))
    expected_no_shows = patient["no_show"].cumsum().shift(fill_value=0).tolist()
    assert patient["prior_no_shows"].tolist() == expected_no_shows


def test_derived_columns():
    raw = pd.DataFrame([{
        "start_time": "2026-10-19T09:30", "booked_at": "2026-10-09T09:30",
        "prior_visits": 4, "prior_no_shows": 1,
    }])
    out = add_derived_columns(raw).iloc[0]
    assert out["lead_time_days"] == 10
    assert out["weekday"] == "0" and out["hour"] == "9"
    assert out["prior_no_show_rate"] == 0.25
    assert out["is_first_visit"] == 0


# --- model -----------------------------------------------------------------


def test_training_writes_metrics_and_beats_baseline(model_path):
    metrics = json.loads(model_path.with_name("metrics.json").read_text())
    chosen = metrics["results"][metrics["chosen_model"]]
    baseline = metrics["results"]["baseline_constant"]
    assert chosen["roc_auc"] > 0.6
    assert chosen["brier"] < baseline["brier"]


def test_probabilities_are_valid(conn, model_path):
    predictor = NoShowPredictor(model_path)
    preds = predictor.predict(load_booked(conn, day=MONDAY.isoformat()))
    assert preds
    assert all(0 <= p.probability <= 1 for p in preds)
    assert all(p.risk == risk_band(p.probability) for p in preds)


def test_model_learns_known_signal(model_path):
    """The generator makes long lead times and past no-shows risky; the model should agree."""
    predictor = NoShowPredictor(model_path)
    base = dict(age=40, distance_km=5.0, prior_visits=5, prior_no_shows=0, prior_no_show_rate=0.0,
                sms_reminder=1, is_first_visit=0, weekday="2", hour="11", department="General Medicine",
                lead_time_days=1.0)
    risky = {**base, "lead_time_days": 40.0, "prior_no_shows": 4, "prior_no_show_rate": 0.8, "sms_reminder": 0}
    low, high = predictor.predict(pd.DataFrame([base, risky])[FEATURES])
    assert high.probability > low.probability
    assert any("missed 4 of 5" in r for r in high.reasons)


def test_missing_model_raises_clear_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="python -m ml.train"):
        NoShowPredictor(tmp_path / "nope.joblib")


@pytest.mark.parametrize("p,band", [(0.05, "low"), (0.199, "low"), (0.2, "medium"), (0.39, "medium"), (0.4, "high")])
def test_risk_bands(p, band):
    assert risk_band(p) == band


# --- API -------------------------------------------------------------------


def test_appointment_risk_endpoint(client, conn):
    appt_id = conn.execute("SELECT id FROM appointments WHERE status = 'booked' LIMIT 1").fetchone()[0]
    body = client.get(f"/api/appointments/{appt_id}/no-show-risk").json()
    assert body["appointment_id"] == appt_id
    assert 0 <= body["no_show_probability"] <= 1
    assert body["risk"] in {"low", "medium", "high"}


def test_risk_for_past_or_unknown_appointment_is_404(client, conn):
    past_id = conn.execute("SELECT id FROM appointments WHERE status = 'completed' LIMIT 1").fetchone()[0]
    assert client.get(f"/api/appointments/{past_id}/no-show-risk").status_code == 404
    assert client.get("/api/appointments/999999/no-show-risk").status_code == 404


def test_upcoming_filters_by_risk(client):
    everyone = client.get("/api/appointments/upcoming", params={"date": MONDAY.isoformat()}).json()
    high = client.get("/api/appointments/upcoming", params={"date": MONDAY.isoformat(), "risk": "high"}).json()
    assert everyone and all(a["start_time"].startswith(MONDAY.isoformat()) for a in everyone)
    assert {a["appointment_id"] for a in high} <= {a["appointment_id"] for a in everyone}
    assert all(a["risk"] == "high" for a in high)


VALID = dict(lead_time_days=30, age=25, distance_km=12, prior_visits=4, prior_no_shows=2,
             sms_reminder=False, weekday=0, hour=9, department="Cardiology")


def test_predict_endpoint(client):
    body = client.post("/api/predict", json=VALID).json()
    assert 0 <= body["no_show_probability"] <= 1
    assert body["model"] == "logistic_regression"


@pytest.mark.parametrize("override", [
    {"age": -1}, {"hour": 24}, {"weekday": 7}, {"prior_no_shows": 9},
    {"department": "Astrology"}, {"lead_time_days": "soon"},
])
def test_predict_rejects_invalid_input(client, override):
    assert client.post("/api/predict", json={**VALID, **override}).status_code == 422
