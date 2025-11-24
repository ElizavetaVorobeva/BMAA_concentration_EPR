import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import joblib

from src.config import (
    PROCESSED_DATA_DIR,
    RF_MODEL_PATH,
    N_ESTIMATORS_RF,
    N_SPLITS_CV,
)

def main() -> None:
    # 1. Загружаем объединённую таблицу
    data_path = PROCESSED_DATA_DIR / "concentrations_with_areas.xlsx"
    df = pd.read_excel(data_path)

    # 2. Формируем X и y
    X = df[["Cu_mM", "area_smoothed"]].values
    y = df["BMAA_mM"].values

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    print("Используем объектов:", len(X))

    # 3. Кросс-валидация двух моделей
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=N_ESTIMATORS_RF,
            random_state=42,
            n_jobs=-1,
        ),
    }

    kf = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)

    for name, model in models.items():
        mae_scores = []
        rmse_scores = []
        r2_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae_scores.append(mean_absolute_error(y_test, y_pred))
            rmse_scores.append(float(np.sqrt(np.mean((y_test - y_pred) ** 2))))
            r2_scores.append(r2_score(y_test, y_pred))

        print(f"\nМодель: {name}")
        print(f"  MAE  = {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
        print(f"  RMSE = {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")
        print(f"  R²   = {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

    # 4. Обучаем финальный RandomForest на всех данных и сохраняем веса
    rf = RandomForestRegressor(
    n_estimators=N_ESTIMATORS_RF,
    random_state=42,
    n_jobs=-1,
)


    rf.fit(X, y)

    RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, RF_MODEL_PATH)

    print(f"\nФинальная модель RandomForest сохранена в: {RF_MODEL_PATH}")

if __name__ == "__main__":
    main()
