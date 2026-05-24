from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits


def load_breast_cancer_dataset():
    data = load_breast_cancer()
    return data.data, data.target


def load_digits_dataset():
    data = load_digits()
    return data.data, data.target

DATASET_REGISTRY = {
    "breast_cancer": load_breast_cancer_dataset,
    "digits": load_digits_dataset
}