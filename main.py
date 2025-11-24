# main.py

from src.config import (
    LISA_ZIP_PATH,
    CONC_INPUT_PATH,
    CONC_WITH_AREAS_PATH,
)
from src.data_loading import extract_lisa_zip, load_bmaa_spectra
from src.preprocessing import preprocess_all_spectra
from src.features import compute_areas_table, merge_areas_into_concentrations

def main() -> None:
    # Распаковка Lisa.zip
    lisa_dir = extract_lisa_zip(LISA_ZIP_PATH, LISA_ZIP_PATH.parent)

    # Загрузка спектров
    raw_spectra = load_bmaa_spectra(lisa_dir)

    # Препроцессинг
    spectra_proc = preprocess_all_spectra(raw_spectra)

    # Расчёт площадей
    areas_df = compute_areas_table(spectra_proc)

    # Мержим площади в таблицу концентраций
    merged_df = merge_areas_into_concentrations(
        areas_df,
        conc_path=CONC_INPUT_PATH,
        out_path=CONC_WITH_AREAS_PATH,
    )

    print("Готово! Объединённая таблица сохранена:")
    print(CONC_WITH_AREAS_PATH)
    print(merged_df.head())


if __name__ == "__main__":
    main()

