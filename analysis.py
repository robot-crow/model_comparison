from pathlib import Path
import pandas as pd
import json

def load_experiments(experiments_dir):
    runs = []

    for run_dir in Path(experiments_dir).iterdir():
        if not run_dir.is_dir():
            continue

        metrics_path = run_dir / "metrics.csv"
        meta_path = run_dir / "meta.json"

        if not metrics_path.exists():
            continue

        df = pd.read_csv(metrics_path)
        df["run_id"] = run_dir.name

        if meta_path.exists():
            meta = json.loads(meta_path.read_text())

            for k, v in meta.items():
                df[k] = [v] * len(df)

        runs.append(df)

    return pd.concat(runs, ignore_index=True)


def summarise(experiments_dir):
    df = load_experiments(experiments_dir)

    summary = (
        df.groupby("model")
        .agg({
            "accuracy": "mean",
            "f1": "mean",
            "roc_auc": "mean"
        })
        .sort_values("accuracy", ascending=False)
    )

    return df, summary

if __name__ == "__main__":
    exp_dir = "model_comparison/experiments"

    df, summary = summarise(exp_dir)

    print("\nRAW RUNS\n")
    print(df)

    print("\nSUMMARY\n")
    print(summary)