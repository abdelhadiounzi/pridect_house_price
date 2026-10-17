from utils import metrics


class Evaluator:
    # Évalue un Trainer entraîné sur les données de test

    def evaluate(self, trainer, X_test, y_test):
        y_pred = trainer.predict(X_test)
        return {
            "MAE": metrics.mae(y_test, y_pred),
            "RMSE": metrics.rmse(y_test, y_pred),
            "R2": metrics.r2(y_test, y_pred),
        }
