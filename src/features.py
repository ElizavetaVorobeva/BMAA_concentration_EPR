# src/features.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import (
    CONC_INPUT_PATH,
    CONC_WITH_AREAS_PATH,
)


def compute_areas_table(
    spectra: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Строит таблицу с колонками:
      - date: дата эксперимента (из df.attrs["date"])
      - file: имя файла
      - area_smoothed: площадь под Y_smooth
    """
    rows: List[dict] = []

    for file_name, df in spectra.items():
        date_str = df.attrs.get("date", None)
        area_smooth = np.trapz(df["Y_smooth"], df["X"])
        rows.append(
            {
                "date": date_str,
                "file": file_name,
                "area_smoothed": area_smooth,
            }
        )

    return pd.DataFrame(rows)


def merge_areas_into_concentrations(
    areas_df: pd.DataFrame,
    conc_path: Path = CONC_INPUT_PATH,
    out_path: Path = CONC_WITH_AREAS_PATH,
) -> pd.DataFrame:
    """
    Загружает таблицу с концентрациями (без площадей),
    дописывает в неё колонку area_smoothed по (date, file)
    и сохраняет в новый Excel.

    Ожидается, что и в conc-файле, и в areas_df есть колонки:
    'date' и 'file'.
    """
    if not conc_path.exists():
        raise FileNotFoundError(f"Файл с концентрациями не найден: {conc_path}")

    conc_df = pd.read_excel(conc_path)

    merged = conc_df.merge(
        areas_df,
        on=["date", "file"],
        how="left",
        validate="one_to_one",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(out_path, index=False)

    return merged
