import numpy as np
from core.model import Model
from core.optimizer import GradientDescent


class LinearRegression(Model):
    # Régression linéaire entraînée par descente de gradient

    def __init__(self, lr=0.05, epochs=1000):
        self.optimizer = GradientDescent(lr)
        self.epochs = epochs
        self.w = None
        self.b = None
        self.history = []

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = np.zeros(1)
        for _ in range(self.epochs):
            pred = X @ self.w + self.b
            err = pred - y
            grad_w = (2 / n) * (X.T @ err)
            grad_b = np.array([2 * err.mean()])
            self.optimizer.update([self.w, self.b], [grad_w, grad_b])
            self.history.append(float(np.mean(err ** 2)))
        return self

    def predict(self, X):
        return X @ self.w + self.b
