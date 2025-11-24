from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import LISA_ZIP_PATH, RAW_DATA_DIR


def extract_lisa_zip(
    zip_path: Path = LISA_ZIP_PATH,
    extract_dir: Path | None = None,
) -> Path:
    """
    Распаковывает архив Lisa.zip в указанную директорию (по умолчанию data/raw)
    и возвращает путь к папке Lisa.
    """
    if extract_dir is None:
        extract_dir = RAW_DATA_DIR

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    lisa_dir = extract_dir / "Lisa"
    if not lisa_dir.exists():
        raise FileNotFoundError(f"Папка {lisa_dir} не найдена после распаковки.")

    return lisa_dir


def is_bmaa_cuso4_file(path: Path) -> bool:
    """
    Оставляем только файлы, где:
    - в названии есть 'BMAA'
    - в названии есть 'CuSO4'
    - нет подстроки 'ure' (pure/URE).
    """
    name = path.name.upper()
    return ("BMAA" in name) and ("CUSO4" in name) and ("URE" not in name)


def read_epr_dat(path: Path) -> pd.DataFrame:
    """
    Читает .dat-файл в формате EPRTextMode и возвращает DataFrame с колонками:
      - X: магнитное поле
      - Y: амплитуда сигнала ЭПР
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="ignore")

    lines = text.splitlines()

    # Находим начало табличной части
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("x-coordinate"):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Не найден заголовок 'x-coordinate' в файле {path.name}")

    pairs: List[Tuple[float, float]] = []
    for line in lines[data_start:]:
        line = line.strip().replace(",", ".")
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            x_val = float(parts[0])
            y_val = float(parts[1])
            pairs.append((x_val, y_val))
        except ValueError:
            continue

    if not pairs:
        raise ValueError(f"Нет валидных данных в файле {path.name}")

    df = pd.DataFrame(pairs, columns=["X", "Y"])
    return df


def load_bmaa_spectra(lisa_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Загружает все спектры BMAA+CuSO4 из папки Lisa,
    возвращает словарь: имя файла -> DataFrame(X, Y).

    Дата эксперимента берётся из имени родительской папки:  dd-mm-YYYY.
    В df сохраняется как атрибут df.attrs["date"].
    """
    spectra: Dict[str, pd.DataFrame] = {}

    for path in lisa_dir.rglob("*.dat"):
        if not is_bmaa_cuso4_file(path):
            continue
        df = read_epr_dat(path)
        df.attrs["date"] = path.parent.name
        spectra[path.name] = df

    if not spectra:
        raise RuntimeError("Не найдено ни одного подходящего спектра BMAA+CuSO4.")

    return spectra
