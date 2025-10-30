"""
config.py
---------
Define rutas, constantes y parámetros globales reutilizables por todo el proyecto.
"""

from pathlib import Path

# Directorios principales
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Subdirectorios de datos
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"

# Configuraciones para EDA
FIGURES_DIR = REPORTS_DIR / "figures"
NUMERIC_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
CATEGORICAL_COLS = ["Gender", "CAEC", "CALC", "MTRANS", "NObeyesdad"]
BINARY_COLS = ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
TARGET_COL = "NObeyesdad"

# Archivos específicos de entrada/salida
RAW_FILENAME = "obesity_estimation_modified.csv"
PROCESSED_FILENAME = "obesity_estimation_clean.csv"
TRAIN_FILENAME = "train_prepared.csv"
TEST_FILENAME = "test_prepared.csv"
PREPROCESSOR_FILENAME = "preprocessor.pkl"
CLASS_DIST_FILENAME = "class_distribution.csv"

# Paths completos
RAW_FILEPATH = DATA_RAW / RAW_FILENAME
PROCESSED_FILEPATH = DATA_PROCESSED / PROCESSED_FILENAME
TRAIN_FILEPATH = DATA_INTERIM / TRAIN_FILENAME
TEST_FILEPATH = DATA_INTERIM / TEST_FILENAME
PREPROCESSOR_FILEPATH = DATA_INTERIM / PREPROCESSOR_FILENAME
CLASS_DIST_FILEPATH = DATA_INTERIM / CLASS_DIST_FILENAME

# Datos para MLFlow
MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "Obesity_Classification"

# Parámetros globales
RANDOM_STATE = 42
LOG_LEVEL = "INFO"
