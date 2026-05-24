# model_comparison/runner.py

import json

import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

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

from model_comparison.datasets import DATASET_REGISTRY

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

def create_experiment_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = EXPERIMENTS_DIR / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path

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
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted")
    }

    if y_proba is not None:
        y_proba = np.array(y_proba)

        is_binary = len(np.unique(y_true)) == 2

        if is_binary:
            if y_proba.ndim == 2:
                y_score = y_proba[:, 1]
            else:
                y_score = y_proba

            results["roc_auc"] = roc_auc_score(y_true, y_score)

        else:
            results["roc_auc"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted"
            )

    return results

def run_experiment():
    """
    Minimal reproducible experiment: compare models across multiple datasets.
    """
    # config
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    print("Starting experiment...")

    results = {}

    # datasets loop (NEW)
    for dataset_spec in config["datasets"]:
        dataset_name = dataset_spec["name"]

        print(f"Loading dataset: {dataset_name}")
        X, y = DATASET_REGISTRY[dataset_name]()

        # split
        print("Splitting dataset.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        results[dataset_name] = {}

        for spec in config["models"]:
            name = spec["name"]
            model_type = spec["type"]
            params = spec.get("params", {})

            model = build_model(model_type, params)

            print(f"Training {name} on data of size {len(X_train)}")

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
            else:
                y_proba = None

            metrics = evaluate_classification(y_test, preds, y_proba)

            results[dataset_name][name] = {
                "model": name,
                **metrics
            }

    print("Experiment complete.")

    # flatten for display
    flat_rows = []
    for dataset_name, model_dict in results.items():
        for model_name, metrics in model_dict.items():
            flat_rows.append({
                "dataset": dataset_name,
                "model": model_name,
                **metrics
            })

    df = pd.DataFrame(flat_rows)
    df = df.sort_values(["dataset", "accuracy"], ascending=[True, False])

    print(df)

    # save run data
    exp_dir = create_experiment_dir()

    # config
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # metadata
    meta = {
        "timestamp": exp_dir.name,
        "datasets": [d["name"] for d in config["datasets"]]
    }

    with open(exp_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    # results table
    df.to_csv(exp_dir / "metrics.csv", index=False)

    return results