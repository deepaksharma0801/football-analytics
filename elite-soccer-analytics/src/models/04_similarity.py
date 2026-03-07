import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

try:
    import umap
except Exception:  # pragma: no cover
    umap = None

try:
    import hdbscan
except Exception:  # pragma: no cover
    hdbscan = None


RANDOM_SEED = 42


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("player_similarity")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def reduce_dimensions(X: np.ndarray, n_components: int = 10) -> np.ndarray:
    if umap is not None:
        reducer = umap.UMAP(n_components=n_components, random_state=RANDOM_SEED)
        return reducer.fit_transform(X)
    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=RANDOM_SEED)
    return pca.fit_transform(X)


def cluster_roles(X: np.ndarray, default_k: int = 8) -> np.ndarray:
    if hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        labels = clusterer.fit_predict(X)
        # fallback for noise-heavy clustering
        if (labels == -1).mean() > 0.5:
            kmeans = KMeans(n_clusters=default_k, random_state=RANDOM_SEED, n_init="auto")
            labels = kmeans.fit_predict(X)
        return labels
    kmeans = KMeans(n_clusters=default_k, random_state=RANDOM_SEED, n_init="auto")
    return kmeans.fit_predict(X)


def main() -> None:
    parser = argparse.ArgumentParser(description="Player similarity system")
    parser.add_argument("--player-features", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/player_embeddings.parquet")
    parser.add_argument("--player-id", type=int, default=None)
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    logger = setup_logger()

    df = pd.read_csv(args.player_features)
    feature_cols = [c for c in df.columns if c not in ["player_id", "role_cluster"]]
    X = df[feature_cols].fillna(0).values

    emb = reduce_dimensions(X, n_components=10)
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    emb_df["player_id"] = df["player_id"].values

    if "role_cluster" in df.columns:
        emb_df["role_cluster"] = df["role_cluster"].values
        subcluster_labels = np.full(len(emb_df), -1, dtype=int)
        for role in emb_df["role_cluster"].unique():
            subset_idx = emb_df[emb_df["role_cluster"] == role].index
            labels = cluster_roles(emb[subset_idx], default_k=4)
            subcluster_labels[subset_idx] = labels
        emb_df["role_subcluster"] = subcluster_labels
    else:
        role_labels = cluster_roles(emb)
        emb_df["role_cluster"] = role_labels

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(args.output, index=False)
    logger.info("Embeddings saved: %s", args.output)

    if args.player_id is not None:
        if args.player_id not in emb_df["player_id"].values:
            raise ValueError("player_id not found")
        idx = emb_df.index[emb_df["player_id"] == args.player_id][0]
        sims = cosine_similarity(emb, emb[idx : idx + 1]).ravel()
        emb_df["similarity"] = sims
        top = (
            emb_df.sort_values("similarity", ascending=False)
            .head(args.topk + 1)
            .iloc[1:]
        )
        logger.info("Top %d similar players to %s:\n%s", args.topk, args.player_id, top[["player_id", "similarity"]].to_string(index=False))


if __name__ == "__main__":
    main()
