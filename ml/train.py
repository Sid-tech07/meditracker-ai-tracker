"""Train and compare no-show models, then save the chosen one.

Candidates
  * Logistic regression (scaled numerics + one-hot categoricals)
  * Histogram gradient boosting (sklearn's LightGBM-style trees)

Evaluation uses a time-based split (oldest 80% train, newest 20% test)
because in production we always predict the future from the past; a random
split would leak later patient history into training.

Selection rule: keep logistic regression unless boosting beats it by more
than AUC_MARGIN on ROC-AUC. Interpretable, well-calibrated probabilities are
worth more to a receptionist than a sliver of ranking accuracy.

Usage:  python -m ml.train
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.config import get_settings
from app.db import get_conn
from app.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, load_history

ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "model.joblib"
METRICS_PATH = ML_DIR / "metrics.json"
AUC_MARGIN = 0.02
RANDOM_STATE = 42


def build_logistic_regression() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    # No class_weight="balanced": it would inflate probabilities and break calibration.
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, C=1.0))])


def build_gradient_boosting() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
    ])
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
        categorical_features=list(range(len(NUMERIC_FEATURES), len(FEATURES))),
        early_stopping=True, random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Average |predicted - observed| no-show rate across probability bins, weighted by size."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(y_prob, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "brier": round(brier_score_loss(y_true, y_prob), 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
        "ece": round(expected_calibration_error(y_true, y_prob), 4),
    }


def time_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("start_time")
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def choose(results: dict) -> str:
    lr, gb = results["logistic_regression"]["roc_auc"], results["gradient_boosting"]["roc_auc"]
    return "gradient_boosting" if gb - lr > AUC_MARGIN else "logistic_regression"


def train(db_path: Path | None = None, model_path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH) -> dict:
    with get_conn(db_path) as conn:
        df = load_history(conn)
    train_df, test_df = time_split(df)
    y_test = test_df[TARGET].to_numpy()

    candidates = {
        "logistic_regression": build_logistic_regression(),
        "gradient_boosting": build_gradient_boosting(),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(train_df[FEATURES], train_df[TARGET])
        results[name] = evaluate(y_test, model.predict_proba(test_df[FEATURES])[:, 1])

    base_rate = float(train_df[TARGET].mean())
    results["baseline_constant"] = evaluate(y_test, np.full(len(y_test), base_rate))
    results["baseline_constant"]["roc_auc"] = 0.5

    chosen = choose(results)
    # Refit the winner on all history before saving: more data, same recipe.
    final = candidates[chosen].fit(df[FEATURES], df[TARGET])

    metrics = {
        "chosen_model": chosen,
        "selection_rule": f"logistic regression unless boosting ROC-AUC is > {AUC_MARGIN} higher",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": len(train_df), "n_test": len(test_df), "n_final_fit": len(df),
        "test_period": [test_df["start_time"].iloc[0], test_df["start_time"].iloc[-1]],
        "base_no_show_rate": round(base_rate, 4),
        "results": results,
    }
    joblib.dump({"model": final, "name": chosen, "features": FEATURES, "metrics": metrics}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    metrics = train(get_settings().database_path)
    print(f"Chosen model: {metrics['chosen_model']}  (test period {metrics['test_period'][0]} → {metrics['test_period'][1]})")
    header = f"{'model':<22}{'ROC-AUC':>9}{'PR-AUC':>9}{'Brier':>8}{'LogLoss':>9}{'ECE':>8}"
    print(header)
    for name, m in metrics["results"].items():
        print(f"{name:<22}{m['roc_auc']:>9}{m['pr_auc']:>9}{m['brier']:>8}{m['log_loss']:>9}{m['ece']:>8}")
    print(f"Saved {MODEL_PATH.relative_to(ML_DIR.parent)} and {METRICS_PATH.relative_to(ML_DIR.parent)}")


if __name__ == "__main__":
    main()
