from pathlib import Path
import pandas as pd
import pytest


DATA_DIR = Path("data/processed")


def test_csv_creation():
    required = ["matches.csv", "events.csv", "players.csv", "teams.csv", "lineups.csv"]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        pytest.skip("Processed CSVs not found")
    for f in required:
        assert (DATA_DIR / f).exists()


def test_no_null_match_id():
    events_path = DATA_DIR / "events.csv"
    if not events_path.exists():
        pytest.skip("events.csv not found")
    df = pd.read_csv(events_path)
    assert df["match_id"].isna().sum() == 0
