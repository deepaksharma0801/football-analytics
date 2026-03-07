#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python3 src/etl/01_download_statsbomb.py \
  --competition 43 --season 3 --max-matches 5

python3 src/etl/02_normalize_events.py \
  --input-root data/raw/statsbomb --output-root data/processed

python3 src/models/02_train_xg.py \
  --events data/processed/events.csv --matches data/processed/matches.csv

python3 src/features/03_player_features.py \
  --events data/processed/events.csv --matches data/processed/matches.csv

python3 src/models/04_similarity.py \
  --player-features data/processed/player_features_rolling12mo.csv

python3 src/models/05_value_ranking.py \
  --player-features data/processed/player_features_rolling12mo.csv

python3 src/models/06_explainability.py \
  --model models/xg_model.joblib --events data/processed/events.csv --player-id 1

echo "Demo pipeline completed."
