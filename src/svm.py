# Linear SVM from scratch (hinge loss + SGD)
import numpy as np

class SVMScratch:
    def __init__(self, lr=0.01, C=1.0, n_iter=1000):
        self.lr = lr
        self.C = C
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        # TODO: training dengan hinge loss (y di {-1, +1})
        pass

    def decision_function(self, X):
        # TODO: hitung Xw + b
        pass

    def predict(self, X):
        # TODO: sign(decision_function)
        pass

