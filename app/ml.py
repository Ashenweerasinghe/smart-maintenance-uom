"""Machine-failure prediction: loads the trained sklearn pipeline and metadata."""
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Order of columns the pipeline was trained on.
FEATURE_ORDER = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Type",
]


@lru_cache(maxsize=1)
def _load():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata


def get_metadata() -> Dict[str, Any]:
    return _load()[1]


def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single prediction. `features` keys must match FEATURE_ORDER."""
    model, metadata = _load()
    row = {k: features[k] for k in FEATURE_ORDER}
    X = pd.DataFrame([row], columns=FEATURE_ORDER)

    proba = float(model.predict_proba(X)[0, 1])
    threshold = float(metadata.get("decision_threshold", 0.5))
    label = int(proba >= threshold)

    if proba >= 0.66:
        risk = "High"
    elif proba >= threshold:
        risk = "Elevated"
    elif proba >= threshold / 2:
        risk = "Moderate"
    else:
        risk = "Low"

    return {
        "prediction": label,
        "prediction_label": metadata["classes"][str(label)],
        "failure_probability": round(proba, 4),
        "risk_level": risk,
        "decision_threshold": threshold,
    }
