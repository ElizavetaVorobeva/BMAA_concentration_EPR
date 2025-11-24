from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold

from .config import (
    N_ESTIMATORS_RF,
    N_SPLITS_CV,
    RF_MODEL_PATH,
    CV_RESULTS_PATH,
)


def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS_CV,
) -> pd.DataFrame:
    """
    Сравнивает две модели (LinearRegression и RandomForest) через k-fold CV.

    Возвращает DataFrame с метриками:
      - MAE_mean, RMSE_mean, R2_mean
      - MAE_std,  RMSE_std,  R2_std
    """
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=N_ESTIMATORS_RF,
            random_state=42,
            n_jobs=-1,
        ),
    }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results: Dict[str, Dict[str, float]] = {}

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
            rmse_scores.append(
                float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            )
            r2_scores.append(r2_score(y_test, y_pred))

        results[name] = {
            "MAE_mean": np.mean(mae_scores),
            "RMSE_mean": np.mean(rmse_scores),
            "R2_mean": np.mean(r2_scores),
            "MAE_std": np.std(mae_scores),
            "RMSE_std": np.std(rmse_scores),
            "R2_std": np.std(r2_scores),
        }

    results_df = pd.DataFrame(results).T
    return results_df


def train_final_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    model_path: Path = RF_MODEL_PATH,
) -> RandomForestRegressor:
    """
    Обучает финальную модель RandomForestRegressor на всех данных
    и сохраняет её в файл.
    """
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS_RF,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, model_path)
    return rf


def save_cv_results(results_df: pd.DataFrame, path: Path = CV_RESULTS_PATH) -> None:
    """
    Сохраняет результаты кросс-валидации в CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=True)
