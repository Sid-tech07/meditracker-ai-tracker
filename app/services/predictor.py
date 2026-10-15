"""Serve the trained no-show model: probabilities, risk bands and reasons."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.features import FEATURES

# Bands chosen around the ~16% base rate: "low" is below average, "high" is roughly
# double the average risk (about 1 in 8 upcoming bookings, a workable call list).
LOW_MAX = 0.15
MEDIUM_MAX = 0.30

_READABLE = {
    "lead_time_days": "booked far in advance",
    "prior_no_shows": "has missed appointments before",
    "prior_no_show_rate": "high personal no-show rate",
    "prior_visits": "visit history",
    "is_first_visit": "first visit",
    "sms_reminder": "no SMS reminder",
    "distance_km": "lives far from clinic",
    "age": "younger patient",
    "weekday": "day of week",
    "hour": "time of day",
    "department": "department",
}


class ModelNotAvailableError(RuntimeError):
    pass


def risk_band(probability: float) -> str:
    if probability < LOW_MAX:
        return "low"
    if probability < MEDIUM_MAX:
        return "medium"
    return "high"


@dataclass(frozen=True)
class Prediction:
    probability: float
    risk: str
    reasons: tuple[str, ...]


class NoShowPredictor:
    def __init__(self, model_path: Path):
        if not Path(model_path).exists():
            raise ModelNotAvailableError(f"No trained model at {model_path}. Run: python -m ml.train")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.name: str = bundle["name"]
        self.metrics: dict = bundle["metrics"]
        if bundle["features"] != FEATURES:
            raise ModelNotAvailableError("Saved model was trained on different features. Retrain it.")

    def predict(self, features: pd.DataFrame) -> list[Prediction]:
        if features.empty:
            return []
        probs = self.model.predict_proba(features[FEATURES])[:, 1]
        reasons = self._reasons(features)
        return [
            Prediction(round(float(p), 4), risk_band(float(p)), r)
            for p, r in zip(probs, reasons, strict=True)
        ]

    def _reasons(self, features: pd.DataFrame, top_k: int = 3) -> list[tuple[str, ...]]:
        """Top features pushing risk *up*, from logistic-regression contributions.

        contribution = coefficient x transformed value, summed per original
        feature (one-hot columns folded back). Boosting has no such cheap,
        exact decomposition, so it returns no reasons.
        """
        if self.name != "logistic_regression":
            return [()] * len(features)
        pre, clf = self.model.named_steps["pre"], self.model.named_steps["clf"]
        transformed = pre.transform(features[FEATURES])
        transformed = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        contrib = transformed * clf.coef_[0]

        names = pre.get_feature_names_out()
        groups: dict[str, list[int]] = {}
        for i, col in enumerate(names):
            raw = col.split("__", 1)[1]
            base = next(f for f in FEATURES if raw == f or raw.startswith(f + "_"))
            groups.setdefault(base, []).append(i)

        out = []
        for row, (_, values) in zip(contrib, features.iterrows(), strict=True):
            per_feature = {f: row[idx].sum() for f, idx in groups.items()}
            ranked = sorted(per_feature.items(), key=lambda kv: kv[1], reverse=True)
            labels: list[str] = []
            for feat, value in ranked:
                if value <= 0.1 or len(labels) == top_k:
                    break
                label = _label(feat, values)
                if label not in labels:  # prior_no_shows and its rate read the same
                    labels.append(label)
            out.append(tuple(labels))
        return out


_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _label(feature: str, values: pd.Series) -> str:
    if feature == "weekday":
        return f"{_WEEKDAYS[int(values['weekday'])]} appointment"
    if feature == "hour":
        return f"{int(values['hour']):02d}:00 slot"
    if feature == "department":
        return f"{values['department']} department"
    if feature == "lead_time_days":
        return f"booked {values['lead_time_days']:.0f} days ahead"
    if feature in ("prior_no_shows", "prior_no_show_rate"):
        return f"missed {int(values['prior_no_shows'])} of {int(values['prior_visits'])} past visits"
    return _READABLE[feature]
