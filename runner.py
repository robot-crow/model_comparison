# model_comparison/runner.py

def run_experiment():
    """
    Minimal vertical slice of the system.
    """

    print("Starting experiment...")

    # fake dataset
    X = [[1, 2], [3, 4]]
    y = [0, 1]

    # fake model
    model_name = "dummy_model"

    # fake training
    print(f"Training {model_name} on data of size {len(X)}")

    # fake prediction
    preds = [0, 1]

    # fake evaluation
    accuracy = 1.0

    result = {
        "model": model_name,
        "accuracy": accuracy
    }

    print("Experiment complete.")
    return result