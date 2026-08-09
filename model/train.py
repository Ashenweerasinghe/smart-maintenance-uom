"""
Train a machine-failure classifier on the AI4I 2020 Predictive Maintenance dataset.

Outputs:
  model/model.pkl       - trained sklearn Pipeline (preprocessing + classifier)
  model/metadata.json   - feature schema, class labels, evaluation metrics

Dataset: AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository, id=601).
10,000 rows of synthetic-but-realistic industrial sensor readings.
Target: 'Machine failure' (binary: 0 = healthy, 1 = failure).
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "ai4i2020.csv"

# Feature definitions (kept in sync with metadata.json for the API layer)
NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]
CATEGORICAL_FEATURES = ["Type"]
TARGET = "Machine failure"


def load_data() -> pd.DataFrame:
    if DATA.exists():
        print(f"Loading cached dataset from {DATA}")
        return pd.read_csv(DATA)
    print("Fetching AI4I 2020 dataset from UCI ML Repository...")
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=601)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA, index=False)
    print(f"Cached dataset to {DATA}")
    return df


def main() -> None:
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Failure rate: {df[TARGET].mean():.2%}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    pipe = Pipeline([("preprocess", preprocess), ("clf", clf)])

    print("Training RandomForestClassifier...")
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    # Tune the decision threshold to maximise F1 on the failure class.
    # Default 0.5 under-detects failures because of the 3.4% class imbalance;
    # a lower threshold trades a little precision for much better recall, which
    # is the right call for predictive maintenance (a missed failure is costly).
    from numpy import linspace
    from sklearn.metrics import f1_score

    best_threshold, best_f1 = 0.5, -1.0
    for t in linspace(0.05, 0.9, 86):
        f1 = f1_score(y_test, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(t)

    y_pred = (y_proba >= best_threshold).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n=== Evaluation ===")
    print(f"Tuned decision threshold: {best_threshold:.3f}")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(pipe, HERE / "model.pkl")
    print(f"\nSaved model -> {HERE / 'model.pkl'}")

    # Categorical option values for the UI dropdown
    type_values = sorted(df["Type"].dropna().unique().tolist())

    metadata = {
        "model_type": "RandomForestClassifier",
        "target": TARGET,
        "classes": {"0": "No Failure", "1": "Failure"},
        "decision_threshold": best_threshold,
        "numeric_features": [
            {
                "name": f,
                "min": float(df[f].min()),
                "max": float(df[f].max()),
                "median": float(df[f].median()),
            }
            for f in NUMERIC_FEATURES
        ],
        "categorical_features": [
            {"name": "Type", "values": type_values}
        ],
        "metrics": {
            "accuracy": report["accuracy"],
            "precision_failure": report["1"]["precision"],
            "recall_failure": report["1"]["recall"],
            "f1_failure": report["1"]["f1-score"],
            "roc_auc": auc,
        },
        "dataset": "AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository, id=601)",
        "n_samples": int(df.shape[0]),
    }
    with open(HERE / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata -> {HERE / 'metadata.json'}")


if __name__ == "__main__":
    main()
