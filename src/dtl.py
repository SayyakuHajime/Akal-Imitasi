# Implementasi Decision Tree Learning (ID3 / CART) from scratch
import numpy as np

class ID3Classifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X, y):
        # TODO: bangun tree pakai entropy / information gain
        pass

    def predict(self, X):
        # TODO: traversal tree untuk setiap sampel
        pass


class CARTClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    def fit(self, X, y):
        # TODO: bangun tree pakai Gini / MSE
        pass

    def predict(self, X):
        pass

