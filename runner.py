# model_comparison/runner.py

import json

import pandas as pd

from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

def build_model(model_type, params):
    if model_type == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**params))
        ])

    if model_type == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(**params))
        ])

    if model_type == "random_forest":
        return Pipeline([
            ("model", RandomForestClassifier(**params))
        ])

    raise ValueError(f"Unknown model type: {model_type}")

def evaluate_classification(y_true, y_pred, y_proba=None):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

    # ROC-AUC only if probabilities exist
    if y_proba is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_proba)

    return results

def run_experiment():
    """
    Minimal reproducible experiment: compare models on one dataset.
    """

    print("Starting experiment...")

    # dataset
    print("Loading dataset.")
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # split
    print("Splitting dataset.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # models
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    results = {}

    for spec in config["models"]:
        name = spec["name"]
        model_type = spec["type"]
        params = spec.get("params", {})

        model = build_model(model_type, params)

        print(f"Training {name} on data of size {len(X_train)}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None

        metrics = evaluate_classification(y_test, preds, y_proba)

        results[name] = {
            "model": name,
            **metrics
        }

    print("Experiment complete.")
    df = pd.DataFrame(results).T
    df = df.sort_values("accuracy", ascending=False)

    print(df)
    return results