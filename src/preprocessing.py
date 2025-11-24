from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import SMOOTHING_WINDOW


def baseline_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычитает среднее значение амплитуды по спектру (baseline correction).
    Добавляет колонку 'Y_corr'.
    """
    baseline = df["Y"].mean()
    df["Y_corr"] = df["Y"] - baseline
    return df


def normalize_spectrum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормирует спектр так, чтобы max(|Y_corr|) == 1.
    Добавляет колонку 'Y_norm'.
    """
    max_abs = df["Y_corr"].abs().max()
    if max_abs == 0:
        df["Y_norm"] = df["Y_corr"]
    else:
        df["Y_norm"] = df["Y_corr"] / max_abs
    return df


def smooth_spectrum(df: pd.DataFrame, window: int = SMOOTHING_WINDOW) -> pd.DataFrame:
    """
    Применяет скользящее среднее к нормированному спектру.
    Добавляет колонку 'Y_smooth'.
    """
    df["Y_smooth"] = (
        df["Y_norm"]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )
    return df


def preprocess_all_spectra(
    spectra: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Применяет baseline-коррекцию, нормировку и сглаживание
    ко всем спектрам в словаре.
    """
    processed: Dict[str, pd.DataFrame] = {}
    for name, df in spectra.items():
        df_proc = df.copy()
        df_proc = baseline_correction(df_proc)
        df_proc = normalize_spectrum(df_proc)
        df_proc = smooth_spectrum(df_proc)
        processed[name] = df_proc
    return processed
