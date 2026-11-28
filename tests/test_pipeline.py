import os
import numpy as np

from core.dataset import Dataset
from core.linear_regression import LinearRegression
from pipeline.evaluator import Evaluator
from pipeline.trainer import Trainer
from utils import metrics
from utils.preprocessing import StandardScaler


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_standard_scaler_roundtrip_and_constant_feature():
    X = np.array([[10.0, 5.0], [20.0, 5.0], [30.0, 5.0]])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    assert np.allclose(Xs[:, 0], [-1.22474487, 0.0, 1.22474487], atol=1e-6)
    assert np.allclose(Xs[:, 1], [0.0, 0.0, 0.0])
    assert np.allclose(scaler.inverse_transform(Xs), X)


def test_metrics_exact_predictions():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])

    assert metrics.mse(y_true, y_pred) == 0.0
    assert metrics.rmse(y_true, y_pred) == 0.0
    assert metrics.mae(y_true, y_pred) == 0.0
    assert metrics.r2(y_true, y_pred) == 1.0


def test_training_evaluation_and_model_persistence(tmp_path):
    data_path = os.path.join(ROOT, "data", "house_data.csv")
    dataset = Dataset(data_path, target="price")
    X_train, X_test, y_train, y_test = dataset.split(test_size=0.2, seed=42)

    model = LinearRegression(lr=0.05, epochs=1000)
    trainer = Trainer(model, dataset.features).train(X_train, y_train)

    res = Evaluator().evaluate(trainer, X_test, y_test)
    assert res["R2"] > 0.9
    assert res["MAE"] > 0
    assert res["RMSE"] >= res["MAE"]

    model_file = tmp_path / "best_model.pkl"
    trainer.save(str(model_file))
    loaded = Trainer.load(str(model_file))

    assert np.allclose(trainer.predict(X_test), loaded.predict(X_test))
