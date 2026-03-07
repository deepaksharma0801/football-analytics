# Elite Soccer Analytics — Recruitment & Tactical Intelligence

A production‑grade football analytics system built on StatsBomb Open Data. It covers the full pipeline from raw events to models, player embeddings, undervalued player ranking, and a Streamlit dashboard that’s easy to read for non‑technical users.

## What This Project Delivers
- Automated data pipeline (download → normalize → validate)
- Custom xG model
- Player features and role clustering
- Similarity search for player scouting
- Undervalued player shortlist
- Streamlit dashboard with clear, human‑readable visuals
- Tests, CI, and Docker

## Quick Start (End‑to‑End)
```bash
cd /Users/snadimi3/Documents/Football/elite-soccer-analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/run_demo.sh
