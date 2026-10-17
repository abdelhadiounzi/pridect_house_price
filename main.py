import os
from core.dataset import Dataset
from core.linear_regression import LinearRegression
from pipeline.trainer import Trainer
from pipeline.evaluator import Evaluator

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    # 1. Chargement et préparation des données
    dataset = Dataset(os.path.join(ROOT, "data", "house_data.csv"), target="price")
    X_train, X_test, y_train, y_test = dataset.split()

    # 2. Entraînement de la régression linéaire
    model = LinearRegression(lr=0.05, epochs=1000)
    trainer = Trainer(model, dataset.features).train(X_train, y_train)

    # 3. Évaluation
    evaluator = Evaluator()
    res = evaluator.evaluate(trainer, X_test, y_test)
    print(f"Régression linéaire   MAE={res['MAE']:.0f}  RMSE={res['RMSE']:.0f}  R2={res['R2']:.4f}")

    # 4. Sauvegarde du modèle
    save_path = os.path.join(ROOT, "models", "best_model.pkl")
    trainer.save(save_path)
    print(f"\nModèle sauvegardé dans {save_path}")


if __name__ == "__main__":
    main()