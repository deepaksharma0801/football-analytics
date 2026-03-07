import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("xg_explainability")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def extract_shots(events: pd.DataFrame) -> pd.DataFrame:
    shots = events[events["event_type"] == "Shot"].copy()
    if "freeze_frame_features" not in shots.columns:
        shots["freeze_frame_features"] = "{}"
    gk_dist = []
    for val in shots["freeze_frame_features"].fillna("{}").values:
        try:
            d = json.loads(val)
            gk_dist.append(d.get("freeze_goalkeeper_dist"))
        except json.JSONDecodeError:
            gk_dist.append(None)
    shots["gk_distance"] = gk_dist
    shots["pressure"] = shots["pressure_flag"].fillna(False) if "pressure_flag" in shots.columns else False
    return shots


def engineer_features(shots: pd.DataFrame) -> pd.DataFrame:
    goal_x, goal_y = 120, 40
    dx = goal_x - shots["x"]
    dy = (goal_y - shots["y"]).abs()
    shots = shots.copy()
    shots["distance_to_goal"] = np.sqrt(dx**2 + dy**2)
    shots["angle_to_goal"] = np.arctan2(7.32 * dx, dx**2 + dy**2)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain xG model using SHAP")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--events", type=str, required=True)
    parser.add_argument("--player-id", type=int, default=None)
    parser.add_argument("--cache", type=str, default="outputs/shap_explainer.joblib")
    args = parser.parse_args()

    logger = setup_logger()

    pipeline = joblib.load(args.model)
    events = pd.read_csv(args.events)
    shots = extract_shots(events)
    shots = engineer_features(shots)

    if shots.empty:
        raise ValueError("No shots found for explainability")

    X = shots.copy()

    pre = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]
    X_trans = pre.transform(X)
    feature_names = []
    if hasattr(pre, "get_feature_names_out"):
        feature_names = pre.get_feature_names_out()
        feature_names = [name.replace("num__", "").replace("cat__", "").replace("_", " ") for name in feature_names]

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        explainer = joblib.load(cache_path)
    else:
        explainer = shap.TreeExplainer(model)
        joblib.dump(explainer, cache_path)

    shap_values = explainer.shap_values(X_trans)
    expected_value = explainer.expected_value
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if isinstance(expected_value, list):
        expected_value = expected_value[1]

    # summary plot
    plt.figure(figsize=(10, 6))
    if feature_names:
        if hasattr(X_trans, "toarray"):
            X_plot = X_trans.toarray()
        else:
            X_plot = X_trans
        X_plot = pd.DataFrame(X_plot, columns=feature_names)
        shap.summary_plot(shap_values, X_plot, show=False)
    else:
        shap.summary_plot(shap_values, X_trans, show=False)
    summary_path = Path("outputs/shap_summary.png")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150)
    plt.close()
    logger.info("SHAP summary saved: %s", summary_path)

    if args.player_id is not None and "player_id" in shots.columns:
        player_rows = shots[shots["player_id"] == args.player_id]
        if not player_rows.empty:
            row_idx = player_rows.index[0]
            plt.figure(figsize=(8, 4))
            shap.force_plot(
                expected_value,
                shap_values[row_idx],
                X_trans[row_idx],
                show=False,
                matplotlib=True,
            )
            player_path = Path(f"outputs/shap_player_{args.player_id}.png")
            plt.tight_layout()
            plt.savefig(player_path, dpi=150)
            plt.close()
            logger.info("SHAP player plot saved: %s", player_path)


if __name__ == "__main__":
    main()
