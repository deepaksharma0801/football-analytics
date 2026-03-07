import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve


RANDOM_SEED = 42


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("xg_train")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def engineer_features(shots: pd.DataFrame) -> pd.DataFrame:
    # StatsBomb pitch is 120 x 80
    goal_x, goal_y = 120, 40
    dx = goal_x - shots["x"]
    dy = (goal_y - shots["y"]).abs()
    shots = shots.copy()
    shots["distance_to_goal"] = np.sqrt(dx**2 + dy**2)
    shots["angle_to_goal"] = np.arctan2(7.32 * dx, dx**2 + dy**2)  # 7.32m goal width approx

    # categorical
    if "body_part" not in shots.columns:
        shots["body_part"] = "Unknown"
    if "shot_type" not in shots.columns:
        shots["shot_type"] = "Unknown"
    if "assist_type" not in shots.columns:
        shots["assist_type"] = "Unknown"
    if "pressure" not in shots.columns:
        shots["pressure"] = False

    shots["body_part"] = shots["body_part"].fillna("Unknown")
    shots["shot_type"] = shots["shot_type"].fillna("Unknown")
    shots["assist_type"] = shots["assist_type"].fillna("Unknown")
    shots["pressure"] = shots["pressure"].fillna(False)
    med = shots["gk_distance"].median()
    shots["gk_distance"] = shots["gk_distance"].fillna(med if not np.isnan(med) else 0)

    return shots


def extract_shots(events: pd.DataFrame) -> pd.DataFrame:
    shots = events[events["event_type"] == "Shot"].copy()
    # Create additional columns from freeze_frame_features if available
    gk_dist = []
    if "freeze_frame_features" not in shots.columns:
        shots["freeze_frame_features"] = "{}"
    for val in shots["freeze_frame_features"].fillna("{}").values:
        try:
            d = json.loads(val)
            gk_dist.append(d.get("freeze_goalkeeper_dist"))
        except json.JSONDecodeError:
            gk_dist.append(None)
    shots["gk_distance"] = gk_dist

    # use normalized columns if present
    shots["shot_type"] = shots.get("shot_type")
    shots["assist_type"] = shots.get("assist_type")
    shots["pressure"] = shots["pressure_flag"].fillna(False)

    return shots


def build_pipeline() -> Pipeline:
    numeric_features = ["distance_to_goal", "angle_to_goal", "x", "y", "gk_distance"]
    categorical_features = ["body_part", "shot_type", "assist_type", "pressure"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
    )

    return Pipeline(steps=[("pre", preprocessor), ("model", model)])


def time_series_folds(match_ids: np.ndarray, n_splits: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(match_ids):
        yield match_ids[train_idx], match_ids[test_idx]


def evaluate_cv(df: pd.DataFrame, pipeline: Pipeline, logger: logging.Logger) -> Dict[str, float]:
    match_ids = np.array(sorted(df["match_id"].unique()))
    aucs = []
    briers = []
    loglosses = []

    if len(match_ids) < 5:
        split = int(len(match_ids) * 0.8) if len(match_ids) > 1 else 1
        train_matches = match_ids[:split]
        test_matches = match_ids[split:] if split < len(match_ids) else match_ids[:split]
        folds = [(train_matches, test_matches)]
    else:
        folds = list(time_series_folds(match_ids))

    for train_matches, test_matches in folds:
        train = df[df["match_id"].isin(train_matches)]
        test = df[df["match_id"].isin(test_matches)]

        X_train = train
        y_train = train["is_goal"]
        X_test = test
        y_test = test["is_goal"]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict_proba(X_test)[:, 1]

        aucs.append(roc_auc_score(y_test, preds))
        briers.append(brier_score_loss(y_test, preds))
        loglosses.append(log_loss(y_test, preds))

    metrics = {
        "auc": float(np.mean(aucs)),
        "brier": float(np.mean(briers)),
        "log_loss": float(np.mean(loglosses)),
    }
    logger.info("CV metrics: %s", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom xG model")
    parser.add_argument("--events", type=str, required=True)
    parser.add_argument("--matches", type=str, required=False)
    parser.add_argument("--model-out", type=str, default="models/xg_model.joblib")
    parser.add_argument("--metrics-out", type=str, default="models/xg_metrics.json")
    args = parser.parse_args()

    logger = setup_logger()

    events = pd.read_csv(args.events)
    shots = extract_shots(events)

    # label
    if "shot_outcome" in shots.columns:
        shots["is_goal"] = shots["shot_outcome"].fillna("").eq("Goal").astype(int)
    else:
        shots["is_goal"] = 0

    # features
    shots = engineer_features(shots)

    pipeline = build_pipeline()
    metrics = evaluate_cv(shots, pipeline, logger)

    # fit full model
    X = shots
    y = shots["is_goal"]
    pipeline.fit(X, y)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # calibration curve saved as part of explainability module; here we just compute
    prob_true, prob_pred = calibration_curve(y, pipeline.predict_proba(X)[:, 1], n_bins=10)
    logger.info("Calibration bins: %s", list(zip(prob_pred, prob_true)))


if __name__ == "__main__":
    main()
