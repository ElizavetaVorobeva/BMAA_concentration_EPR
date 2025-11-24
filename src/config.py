# src/config.py
from pathlib import Path
import numpy as np
import random

# Фиксируем сиды для воспроизводимости
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Базовые пути (относительно корня проекта)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "plots"

# Имена файлов по умолчанию
LISA_ZIP_PATH = RAW_DATA_DIR / "Lisa.zip"
AREAS_WITH_CONC_PATH = PROCESSED_DATA_DIR / "areas_with_conc.xlsx"
CONC_INPUT_PATH = PROCESSED_DATA_DIR / "concentrations.xlsx"           # файл БЕЗ площадей
CONC_WITH_AREAS_PATH = PROCESSED_DATA_DIR / "concentrations_with_areas.xlsx"  # файл С площадями
CV_RESULTS_PATH = PROCESSED_DATA_DIR / "cv_results.csv"
RF_MODEL_PATH = WEIGHTS_DIR / "rf_bmaa_model.joblib"

# Параметры сглаживания
SMOOTHING_WINDOW = 7

# Параметры кросс-валидации
N_SPLITS_CV = 5
N_ESTIMATORS_RF = 300
