# Model Comparison Framework

A lightweight, config-driven framework for running and comparing machine learning models across multiple datasets, with persistent experiment tracking and simple post-run analysis.

## Why this exists

This project is a small, controlled framework for running repeatable ML experiments and comparing models across datasets in a structured way.

## What this is

This project is a minimal experiment runner for classification models.

### It allows you to:

- define models and datasets in a config file
- run all combinations consistently
- save results per run
- analyse results across multiple runs

It is designed to be readable, reproducible, and easy to extend without external experiment tracking tools.

## Project structure:

```
model_comparison/
├── runner.py              # Executes experiments
├── analysis.py            # Aggregates results across runs
├── datasets.py            # Dataset registry
├── config.json            # Experiment configuration
├── experiments/           # Output directory (auto-generated)
│   └── <timestamp>/
│       ├── config.json
│       ├── meta.json
│       └── metrics.csv
```

## How it works
### 1. Define experiment config

Models and datasets are defined in config.json.

Example:

```JSON
{
  "datasets": [
    { "name": "digits" },
    { "name": "breast_cancer" }
  ],
  "models": [
    {
      "name": "logistic_regression",
      "type": "logistic_regression",
      "params": { "max_iter": 500 }
    },
    {
      "name": "random_forest",
      "type": "random_forest",
      "params": {}
    }
  ]
}
```

### 2. Run experiments

#### Running it
```Bash
python -m model_comparison
```

Each run:

- loads datasets from registry
- trains all configured models
- evaluates predictions
- saves results to a timestamped folder

Output per run:

```
experiments/
└── 2026-05-24_16-58-20/
    ├── config.json
    ├── meta.json
    └── metrics.csv
```

### 3. Analyse results

#### Run:

```Bash
python model_comparison/analysis.py
```

This:

- loads all experiment runs
- combines results into a single dataframe
- produces summary statistics per model

### Outputs:

raw per-run results
aggregated model performance summary
Metrics tracked

### For each model:
- accuracy
- precision (weighted)
- recall (weighted)
- f1 score (weighted)
- ROC-AUC (binary and multiclass supported)


## Limitations
- No hyperparameter search framework
- No distributed execution
- No experiment tagging or filtering system
- No visualisation layer yet
- No database-backed tracking (file-based only)

