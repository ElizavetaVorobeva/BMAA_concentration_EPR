import sys
from pathlib import Path

# добавляем корень проекта в пути поиска модулей
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from src.config import PROCESSED_DATA_DIR, CONC_WITH_AREAS_PATH

def main():
    df = pd.read_excel(CONC_WITH_AREAS_PATH)

    # Проверим, что нужные столбцы есть
    required = ["V All", "V Cu", "V BMAA", "Cu start", "BMAA start"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Нет столбца {col} в таблице!")

    # Расчёт концентраций
    df["Cu_mM"] = df["Cu start"] * (df["V Cu"] / df["V All"])
    df["BMAA_mM"] = df["BMAA start"] * (df["V BMAA"] / df["V All"])

    df.to_excel(CONC_WITH_AREAS_PATH, index=False)
    print("Концентрации Cu_mM и BMAA_mM добавлены.")

if __name__ == "__main__":
    main()
