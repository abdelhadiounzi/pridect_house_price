import os
import pickle
from utils.preprocessing import StandardScaler


class Trainer:
    # Normalise les données, entraîne le modèle et peut le sauvegarder

    def __init__(self, model, features):
        self.model = model
        self.features = features
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def train(self, X_train, y_train):
        Xs = self.x_scaler.fit_transform(X_train)
        ys = self.y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        self.model.fit(Xs, ys)
        return self

    def predict(self, X):
        Xs = self.x_scaler.transform(X)
        ys = self.model.predict(Xs)
        return self.y_scaler.inverse_transform(ys.reshape(-1, 1)).ravel()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
