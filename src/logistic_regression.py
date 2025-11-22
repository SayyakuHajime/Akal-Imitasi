# Logistic Regression from scratch
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.01, n_iter=1000, l2=0.0):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.w = None
        self.b = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # TODO: implementasi gradient descent
        pass

    def predict_proba(self, X):
        # TODO: return probabilitas kelas positif
        pass

    def predict(self, X, threshold=0.5):
        # TODO: return label 0/1
        pass

