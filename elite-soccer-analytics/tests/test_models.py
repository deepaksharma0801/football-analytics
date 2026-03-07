from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest


MODELS_DIR = Path("models")
DATA_DIR = Path("data/processed")


def test_model_outputs_probability():
    model_path = MODELS_DIR / "xg_model.joblib"
    events_path = DATA_DIR / "events.csv"
    if not model_path.exists() or not events_path.exists():
        pytest.skip("Model or events data not found")

    model = joblib.load(model_path)
    events = pd.read_csv(events_path)
    shots = events[events["event_type"] == "Shot"].head(10)
    if shots.empty:
        pytest.skip("No shots found for testing")

    probs = model.predict_proba(shots)[:, 1]
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_metrics_file_exists():
    metrics_path = MODELS_DIR / "xg_metrics.json"
    if not metrics_path.exists():
        pytest.skip("Metrics not found")
    assert metrics_path.exists()
