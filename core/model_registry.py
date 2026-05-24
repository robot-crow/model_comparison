# model_comparison/core/model_registry.py

class ModelRegistry:
    """
    A registry of models for experimentation.
    """
    def __init__(self):
        self._models = {}

    def register(self, name, model):
        self._models[name] = model

    def get(self, name):
        return self._models[name]