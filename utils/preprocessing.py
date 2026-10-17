import numpy as np


class StandardScaler:
    # Centre et réduit les données : (x - moyenne) / ecart_type

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float) * self.std + self.mean
