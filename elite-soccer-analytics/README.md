# Elite Soccer Analytics — Recruitment & Tactical Intelligence

A production-grade, end-to-end football analytics system built on StatsBomb Open Data. The stack delivers a full data engineering pipeline, event-level modeling, custom xG, role-aware player embeddings, similarity search, undervalued player ranking, explainability, and a Streamlit dashboard.

## Project Overview
This system is designed to resemble a professional analytics platform used by clubs or performance departments. It supports reproducible pipelines, modular modeling, and tactical visualizations.

### Key Capabilities
- Data engineering pipeline (download → normalize → validate)
- Event-level modeling with custom xG
- Role-aware player embeddings + similarity search
- Undervalued player detection
- Explainability via SHAP
- Streamlit dashboard with tactical visuals
- Tests + CI + Dockerized deployment

## Architecture diagram
```
raw data (StatsBomb Open Data)
        ↓
ETL download + validation
        ↓
Event normalization → relational CSVs
        ↓
Feature engineering
        ↓
Custom xG model + explainability
        ↓
Player embeddings + similarity search
        ↓
Undervalued player ranking
        ↓
Streamlit dashboard + PDF reports
```

## Dataset Citation
StatsBomb Open Data is used for event data.
- StatsBomb Open Data: https://github.com/statsbomb/open-data

Optional: External dataset via HuggingFace for market-value proxies
```
from datasets import load_dataset
# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("3zden/fbref_football_player_performance_2024-2025")
```

## Project Structure
```
elite-soccer-analytics/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
├── notebooks/
├── src/
│   ├── etl/
│   ├── features/
│   ├── models/
│   └── visuals/
├── app/
├── tests/
├── scripts/
├── requirements.txt
├── Dockerfile
├── README.md
└── .github/workflows/ci.yml
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## End-to-End Pipeline
```bash
# 1) Download StatsBomb data
python3 src/etl/01_download_statsbomb.py --competition 43 --season 3

# 2) Normalize events
python3 src/etl/02_normalize_events.py --input-root data/raw/statsbomb --output-root data/processed

# 3) Train xG model
python3 src/models/02_train_xg.py --events data/processed/events.csv --matches data/processed/matches.csv

# 4) Player features + roles
python3 src/features/03_player_features.py --events data/processed/events.csv --matches data/processed/matches.csv

# 5) Player similarity embeddings
python3 src/models/04_similarity.py --player-features data/processed/player_features_rolling12mo.csv

# 6) Undervalued player ranking
python3 src/models/05_value_ranking.py --player-features data/processed/player_features_rolling12mo.csv

# 7) Model explainability
python3 src/models/06_explainability.py --model models/xg_model.joblib --events data/processed/events.csv --player-id 1
```

### Minimal Demo Script
```bash
./scripts/run_demo.sh
```

## Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```

## Testing and all
```bash
pytest -q
```

## Docker
```bash
docker build -t elite-soccer-analytics .
docker run -p 8501:8501 elite-soccer-analytics
```

## Skills Demonstrated
- Data engineering and pipeline-streamlining automation
- Event-level ML modeling (custom xG)
- Feature engineering and role clustering
- Embeddings + similarity search
- Explainability and model diagnostics
- End-user analytics dashboard
- CI, testing, and deployment readiness

## Notes
- Rolling 12-month features are approximated from match order when explicit dates are unavailable in open data.
- External market-value proxies can be loaded from HuggingFace if credentials are provided; otherwise dummy values are generated.
