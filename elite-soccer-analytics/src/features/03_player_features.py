import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


RANDOM_SEED = 42


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("player_features")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def progressive_distance(start_x: float, start_y: float, end_x: float, end_y: float) -> float:
    if any(pd.isna(v) for v in [start_x, start_y, end_x, end_y]):
        return 0.0
    return (end_x - start_x)


def compute_player_minutes(events: pd.DataFrame) -> pd.DataFrame:
    # approximate minutes by max minute in each match per player
    mins = (
        events.groupby(["player_id", "match_id"])["minute"].max().reset_index()
    )
    mins["minutes"] = mins["minute"].fillna(0) + 1
    player_minutes = mins.groupby("player_id")["minutes"].sum().reset_index()
    return player_minutes


def build_touch_distribution(events: pd.DataFrame) -> pd.DataFrame:
    # 6x4 grid on 120x80 pitch
    events = events.dropna(subset=["x", "y", "player_id"]).copy()
    x_bins = np.linspace(0, 120, 7)
    y_bins = np.linspace(0, 80, 5)
    events["x_bin"] = np.digitize(events["x"], x_bins) - 1
    events["y_bin"] = np.digitize(events["y"], y_bins) - 1
    events = events[(events["x_bin"] >= 0) & (events["x_bin"] < 6) & (events["y_bin"] >= 0) & (events["y_bin"] < 4)]
    events["bin_id"] = events["x_bin"] * 4 + events["y_bin"]

    touch_counts = (
        events.groupby(["player_id", "bin_id"]).size().unstack(fill_value=0)
    )
    for i in range(24):
        if i not in touch_counts.columns:
            touch_counts[i] = 0
    touch_counts = touch_counts[sorted(touch_counts.columns)]
    touch_dist = touch_counts.div(touch_counts.sum(axis=1).replace(0, 1), axis=0)
    touch_dist.columns = [f"touch_bin_{i}" for i in range(24)]
    touch_dist = touch_dist.reset_index()
    return touch_dist


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute player rolling 12-month features and role clusters")
    parser.add_argument("--events", type=str, required=True)
    parser.add_argument("--matches", type=str, required=False)
    parser.add_argument("--output", type=str, default="data/processed/player_features_rolling12mo.csv")
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    logger = setup_logger()

    events = pd.read_csv(args.events)
    shots = events[events["event_type"] == "Shot"].copy()
    passes = events[events["event_type"] == "Pass"].copy()
    carries = events[events["event_type"] == "Carry"].copy()

    # non-penalty xG
    npxg = shots[shots["shot_type"].fillna("") != "Penalty"].groupby("player_id")["shot_statsbomb_xg"].sum()

    # xA from assisted shot id mapping
    shot_xg_map = shots.set_index("event_id")["shot_statsbomb_xg"].to_dict()
    passes["xA"] = passes["pass_assisted_shot_id"].map(shot_xg_map).fillna(0)
    xa = passes.groupby("player_id")["xA"].sum()

    # progressive passes / carries
    passes["progressive_pass"] = (
        passes.apply(lambda r: progressive_distance(r.get("x"), r.get("y"), r.get("end_x"), r.get("end_y")) > 10, axis=1)
    )
    carries["progressive_carry"] = (
        carries.apply(lambda r: progressive_distance(r.get("x"), r.get("y"), r.get("end_x"), r.get("end_y")) > 10, axis=1)
    )
    progressive_passes = passes.groupby("player_id")["progressive_pass"].sum()
    progressive_carries = carries.groupby("player_id")["progressive_carry"].sum()

    pressures = events[events["event_type"] == "Pressure"].groupby("player_id").size()
    tackles = events[events["event_type"].isin(["Duel", "Tackle"])].groupby("player_id").size()
    interceptions = events[events["event_type"] == "Interception"].groupby("player_id").size()

    passes_final_third = passes[passes["end_x"] >= 80].groupby("player_id").size()

    minutes = compute_player_minutes(events).set_index("player_id")

    # assemble
    metrics = pd.DataFrame({
        "npxg": npxg,
        "xa": xa,
        "progressive_passes": progressive_passes,
        "progressive_carries": progressive_carries,
        "pressures": pressures,
        "tackles": tackles,
        "interceptions": interceptions,
        "passes_final_third": passes_final_third,
    }).fillna(0)

    metrics = metrics.join(minutes, how="left").fillna({"minutes": 0})
    per90_metrics = metrics.drop(columns=["minutes"]).div(metrics["minutes"].replace(0, 1) / 90, axis=0)
    per90_metrics = per90_metrics.add_prefix("per90_")
    per90_metrics["minutes"] = metrics["minutes"]
    per90_metrics["player_id"] = per90_metrics.index
    per90_metrics = per90_metrics.reset_index(drop=True)

    touch_dist = build_touch_distribution(events)

    # PCA on touch distribution
    touch_cols = [c for c in touch_dist.columns if c.startswith("touch_bin_")]
    pca = PCA(n_components=8, random_state=RANDOM_SEED)
    touch_emb = pca.fit_transform(touch_dist[touch_cols]) if len(touch_dist) else np.zeros((0, 8))
    touch_emb_df = pd.DataFrame(touch_emb, columns=[f"touch_pca_{i}" for i in range(8)])
    touch_emb_df["player_id"] = touch_dist["player_id"].values if len(touch_dist) else []

    # merge
    df = per90_metrics.merge(touch_emb_df, on="player_id", how="left")

    # role clustering
    role_features = df.drop(columns=["player_id"], errors="ignore").fillna(0)
    kmeans = KMeans(n_clusters=args.k, random_state=RANDOM_SEED, n_init="auto")
    df["role_cluster"] = kmeans.fit_predict(role_features)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Player features saved: %s", args.output)


if __name__ == "__main__":
    main()
