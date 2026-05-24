# model_comparison/__main__.py

from model_comparison.runner import run_experiment

def main():
    """
    Entry point for running a default experiment.
    """
    result = run_experiment()
    print("Main returned result (raw):")
    print(result)


if __name__ == "__main__":
    main()