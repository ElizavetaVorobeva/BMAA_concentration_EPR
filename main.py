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
  # main.py
    from src.config import LISA_ZIP_PATH, CONC_INPUT_PATH, CONC_WITH_AREAS_PATH
    from src.data_loading import extract_lisa_zip, load_bmaa_spectra
    from src.preprocessing import preprocess_all_spectra
    from src.features import compute_areas_table, merge_areas_into_concentrations

    def main() -> None:
        # 1. Распаковка Lisa.zip
        lisa_dir = extract_lisa_zip(LISA_ZIP_PATH, LISA_ZIP_PATH.parent)

        # 2. Загрузка спектров BMAA+CuSO4
        raw_spectra = load_bmaa_spectra(lisa_dir)

        # 3. Препроцессинг (baseline, нормировка, сглаживание)
        spectra_proc = preprocess_all_spectra(raw_spectra)

        # 4. Расчёт площадей под сглаженными спектрами
        areas_df = compute_areas_table(spectra_proc)

        # 5. Мерджим площади в существующий файл с концентрациями
        merged_df = merge_areas_into_concentrations(
            areas_df,
            conc_path=CONC_INPUT_PATH,
            out_path=CONC_WITH_AREAS_PATH,
        )

        print(f"Готово! Объединённая таблица сохранена в: {CONC_WITH_AREAS_PATH}")
        print(merged_df.head())


    if __name__ == "__main__":
        main()

