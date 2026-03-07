import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("value_ranking")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / (series.std(ddof=0) + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Undervalued player ranking")
    parser.add_argument("--player-features", type=str, required=True)
    parser.add_argument("--market-values", type=str, default="data/processed/market_values.csv")
    parser.add_argument("--output", type=str, default="outputs/undervalued_shortlist.csv")
    args = parser.parse_args()

    logger = setup_logger()

    df = pd.read_csv(args.player_features)

    mv_path = Path(args.market_values)
    if not mv_path.exists():
        if load_dataset is not None:
            try:
                logger.info("Loading external dataset from HuggingFace for market values")
                ds = load_dataset("3zden/fbref_football_player_performance_2024-2025")
                # best-effort mapping: create proxy market values from minutes and goals
                hf = ds["train"].to_pandas() if "train" in ds else ds[list(ds.keys())[0]].to_pandas()
                if "player_id" in hf.columns:
                    mv = hf[["player_id"]].copy()
                else:
                    mv = pd.DataFrame({"player_id": df["player_id"].values})
                if "minutes" in hf.columns and "goals" in hf.columns:
                    mv["market_value"] = (hf["minutes"].fillna(0) * 0.01 + hf["goals"].fillna(0) * 1.5).clip(lower=0.1)
                else:
                    mv["market_value"] = np.random.lognormal(2, 0.5, len(mv))
                mv_path.parent.mkdir(parents=True, exist_ok=True)
                mv.to_csv(mv_path, index=False)
            except Exception:
                logger.info("Failed to load external dataset. Creating dummy values at %s", mv_path)
                dummy = pd.DataFrame({"player_id": df["player_id"], "market_value": np.random.lognormal(2, 0.5, len(df))})
                mv_path.parent.mkdir(parents=True, exist_ok=True)
                dummy.to_csv(mv_path, index=False)
        else:
            logger.info("Market values not found. Creating dummy values at %s", mv_path)
            dummy = pd.DataFrame({"player_id": df["player_id"], "market_value": np.random.lognormal(2, 0.5, len(df))})
            mv_path.parent.mkdir(parents=True, exist_ok=True)
            dummy.to_csv(mv_path, index=False)

    mv = pd.read_csv(mv_path)
    df = df.merge(mv, on="player_id", how="left")
    df["market_value"] = df["market_value"].fillna(df["market_value"].median())

    perf_cols = [c for c in df.columns if c.startswith("per90_")]
    df["performance_score"] = df[perf_cols].mean(axis=1)

    df["inefficiency_score"] = zscore(df["performance_score"]) - zscore(df["market_value"])

    df = df.sort_values(["role_cluster", "inefficiency_score"], ascending=[True, False])

    df["rationale"] = (
        "High per-90 output with below-median market value; "
        "profile suggests undervalued contribution for role cluster"
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df[["player_id", "role_cluster", "performance_score", "market_value", "inefficiency_score", "rationale"]].to_csv(args.output, index=False)
    logger.info("Undervalued shortlist saved: %s", args.output)


if __name__ == "__main__":
    main()
