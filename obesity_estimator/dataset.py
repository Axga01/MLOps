"""
make_dataset.py
---------------
Carga, limpia y guarda la versión procesada del dataset de obesidad.
"""


# ----------------- #
# --- LIBRERÍAS --- #
# ----------------- #
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.config import RAW_FILEPATH, PROCESSED_FILEPATH, LOG_LEVEL


# ------------------------------- #
# --- CONFIGURACIÓN Y LOGGING --- #
# ------------------------------- #
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------- #
# --- FUNCIONES AUXILIARES --- #
# ---------------------------- #
def load_data(input_path: Path) -> pd.DataFrame:
    """Carga el dataset inicial."""
    logger.info(f"Cargando dataset desde {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Dataset cargado con forma {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza general: eliminación de columnas irrelevantes, corrección de tipos, imputación y eliminación de duplicados."""

    # Eliminar columnas no útiles
    # NOTA: esta columna se considera no útil después de inspeccionar su contenido
    df = df.drop(columns=["mixed_type_col"], errors="ignore")

    # Definición de conjuntos de columnas por tipo de dato
    numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    object_cols = ['Gender', 'CAEC', 'CALC', 'MTRANS', 'NObeyesdad']

    # Limpieza de columnas numéricas
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Limpieza de columnas binarias
    binary_map = {'yes': 1, 'no': 0}
    for col in binary_cols:
        df[col] = (
            df[col].astype(str).str.strip().str.lower()
            .replace({'nan': np.nan}).map(binary_map)
        )
    df[binary_cols] = df[binary_cols].astype('Int64')

    # Pueden existir NaN en la variable objetivo. Eliminemos esas filas.
    df = df.dropna(subset=["NObeyesdad"])

    # Limpieza de columnas categóricas
    df = df.replace({'nan': np.nan, '?': np.nan, 'invalid': np.nan, 'error': np.nan})
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Filtrado de outliers "razonables"
    valid_ranges = {
        'Age': (5, 120),
        'Height': (1.3, 2.2),
        'Weight': (30, 200),
        'FCVC': (1, 3),
        'NCP': (1, 3),
        'CH2O': (1, 3),
        'FAF': (0, 5),
        'TUE': (0, 3)
    }
    for col, (min_val, max_val) in valid_ranges.items():
        df.loc[(df[col] < min_val) | (df[col] > max_val), col] = np.nan

    # Imputación de valores nulos según el tipo de columna
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in object_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in binary_cols:
        df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

    # Duplicados
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    logger.info(f"Duplicados eliminados: {before - after}")

    return df


def save_data(df: pd.DataFrame, output_path: Path):
    """Guarda el dataset limpio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset limpio guardado en {output_path}")


# ------------------------- #
# --- PROCESO PRINCIPAL --- #
# ------------------------- #
def main(input_filepath: Path = RAW_FILEPATH, output_filepath: Path = PROCESSED_FILEPATH):
    """Ejecuta el pipeline completo de limpieza de datos."""
    df = load_data(input_filepath)
    df_clean = clean_data(df)
    save_data(df_clean, output_filepath)


# ------------------------- #
# --- EJECUCIÓN DE MAIN --- #
# ------------------------- #
if __name__ == "__main__":
    main()
