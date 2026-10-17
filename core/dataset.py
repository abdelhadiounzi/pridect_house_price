import numpy as np
import pandas as pd


class Dataset:
    

    def __init__(self, path, target="price"):
        self.path = path
        self.target = target
        self.df = pd.read_csv(path)
        self.features = [c for c in self.df.columns if c != target]

    def get_X_y(self):
        X = self.df[self.features].values.astype(float)
        y = self.df[self.target].values.astype(float)
        return X, y

    def split(self, test_size=0.2, seed=42):
        X, y = self.get_X_y()
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n_test = int(len(X) * test_size)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
data = Dataset("data/house_data.csv", target="price")
print(data.get_X_y())