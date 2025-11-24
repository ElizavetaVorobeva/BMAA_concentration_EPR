# Оценка концентрации BMAA по ЭПР-спектрам комплексов BMAA–CuSO₄

## Описание проекта

В работе строится модель машинного обучения, предсказывающая концентрацию BMAA (в мМ)
по ЭПР-спектрам комплексов BMAA–CuSO₄. В качестве признаков используются:

- площадь под сглаженным нормированным ЭПР-спектром (`area_smoothed`);
- концентрация меди в пробе (`Cu_mM`).

Сравниваются две модели:
- LinearRegression (baseline);
- RandomForestRegressor (нелинейная модель).

По результатам 5-кратной кросс-валидации RandomForestRegressor показывает:
MAE ≈ ..., RMSE ≈ ..., R² ≈ ..., что существенно лучше, чем у линейной регрессии.

## Структура репозитория

- `data/raw/Lisa.zip` — архив с ЭПР-спектрами в формате .dat  
- `data/processed/` — обработанные таблицы (площади, концентрации, результаты CV)  
- `src/` — исходный код:
  - `config.py` — глобальные настройки и пути
  - `data_loading.py` — загрузка и парсинг .dat-файлов
  - `preprocessing.py` — baseline-коррекция, нормировка и сглаживание спектров
  - `features.py` — расчёт площадей и концентраций
  - `modeling.py` — обучение моделей и кросс-валидация
- `examples/plots/` — примеры графиков (y_true vs y_pred, residuals, feature importance)
- `weights/` — сохранённые веса модели RandomForest (`rf_bmaa_model.joblib`)
- `main.py` — демонстрационный входной скрипт
- `requirements.txt` — список зависимостей

## Установка

```bash
git clone <url_вашего_репозитория>
cd bmaa-epr-coursework

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
