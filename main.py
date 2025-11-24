from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    LISA_ZIP_PATH,
    AREAS_WITH_CONC_PATH,
)
from src.data_loading import extract_lisa_zip, load_bmaa_spectra
from src.preprocessing import preprocess_all_spectra
from src.features import compute_areas_table, save_areas_with_conc
from src.modeling import (
    cross_validate_models,
    train_final_random_forest,
    save_cv_results,
)


def main() -> None:
    """
    Полный демонстрационный pipeline:
    1. Распаковка и загрузка спектров BMAA+CuSO4.
    2. Препроцессинг (baseline, нормировка, сглаживание).
    3. Расчёт площадей под сглаженными спектрами.
    4. (ВНЕ КОДА) Добавление в таблицу объёмов и исходных концентраций.
    5. Добавление концентраций Cu_mM и BMAA_mM.
    6. Обучение моделей и кросс-валидация.
    """
    # 1. Распаковка Lisa.zip
    lisa_dir = extract_lisa_zip(LISA_ZIP_PATH, LISA_ZIP_PATH.parent)

    # 2. Загрузка спектров
    raw_spectra = load_bmaa_spectra(lisa_dir)

    # 3. Препроцессинг
    spectra_proc = preprocess_all_spectra(raw_spectra)

    # 4. Расчёт площадей под Y_smooth
    areas_df = compute_areas_table(spectra_proc)

    # Здесь ты можешь сохранить areas_df и ДОБАВИТЬ В НЕГО в Excel
    # столбцы V_All, V_Cu, V_BMAA, Cu_start, BMAA_start,
    # после чего считать уже готовый файл areas_with_conc.xlsx.
    save_areas_with_conc(areas_df, AREAS_WITH_CONC_PATH)
    print(f"Таблица с площадями сохранена в {AREAS_WITH_CONC_PATH}")
    print("Теперь добавь объёмы и исходные концентрации в этот файл вручную и перезапусти часть кода для обучения моделей.")


if __name__ == "__main__":
    main()
