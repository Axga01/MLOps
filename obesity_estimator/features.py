"""
preprocess_dataset.py
---------------------
Realiza el preprocesamiento del dataset limpio:
- Divide en train/test
- Aplica escalado y codificación
- Guarda conjuntos preparados y el preprocesador
"""


# ----------------- #
# --- LIBRERÍAS --- #
# ----------------- #
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.config import (
    PROCESSED_FILEPATH,
    TRAIN_FILEPATH,
    TEST_FILEPATH,
    PREPROCESSOR_FILEPATH,
    CLASS_DIST_FILEPATH,
    RANDOM_STATE,
    LOG_LEVEL,
)


# ------------------------------- #
# --- CONFIGURACIÓN Y LOGGING --- #
# ------------------------------- #
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------- #
# --- FUNCIONES AUXILIARES --- #
# ---------------------------- #
def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset desde la ruta indicada."""
    df = pd.read_csv(path)
    if df["NObeyesdad"].isna().any():
        n_missing = df["NObeyesdad"].isna().sum()
        df = df.dropna(subset=["NObeyesdad"])
        logger.warning(f"Se eliminaron {n_missing} filas sin etiqueta NObeyesdad.")
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def split_data(df: pd.DataFrame, target: str, test_size=0.3, random_state=RANDOM_STATE):
    """Divide el dataset en entrenamiento y prueba."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(f"División: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_preprocessor(num_cols, cat_cols):
    """Construye el transformador de columnas (numéricas y categóricas)."""
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    return preprocessor


def save_artifacts(train_df, test_df, preprocessor, class_dist):
    """Guarda los conjuntos procesados, el preprocesador y la distribución de clases."""
    TRAIN_FILEPATH.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_FILEPATH, index=False)
    test_df.to_csv(TEST_FILEPATH, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_FILEPATH)
    class_dist.to_csv(CLASS_DIST_FILEPATH)

    logger.info("Archivos guardados:")
    logger.info(f"- {TRAIN_FILEPATH.name}")
    logger.info(f"- {TEST_FILEPATH.name}")
    logger.info(f"- {PREPROCESSOR_FILEPATH.name}")
    logger.info(f"- {CLASS_DIST_FILEPATH.name}")


# ------------------------- #
# --- PROCESO PRINCIPAL --- #
# ------------------------- #
def main():
    """Ejecución principal del script."""
    df = load_data(PROCESSED_FILEPATH)

    target = "NObeyesdad"
    X_train, X_test, y_train, y_test = split_data(df, target)

    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    logger.info(f"Columnas numéricas: {num_cols}")
    logger.info(f"Columnas categóricas: {cat_cols}")

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # Transformar datos
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols)
    processed_feature_names = num_cols + list(cat_feature_names)

    X_train_df = pd.DataFrame(X_train_processed, columns=processed_feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=processed_feature_names, index=X_test.index)

    train_df = pd.concat([X_train_df, y_train], axis=1)
    test_df = pd.concat([X_test_df, y_test], axis=1)

    logger.info(f"Train final: {train_df.shape}, Test final: {test_df.shape}")

    # Distribución de clases
    ordered_classes = [
        'insufficient_weight', 'normal_weight', 'overweight_level_i', 'overweight_level_ii',
        'obesity_type_i', 'obesity_type_ii', 'obesity_type_iii'
    ]

    train_dist = y_train.value_counts(normalize=True).round(3)
    test_dist = y_test.value_counts(normalize=True).round(3)

    class_dist = pd.DataFrame({'Train': train_dist, 'Test': test_dist}).reindex(ordered_classes)

    # Guardar resultados
    save_artifacts(train_df, test_df, preprocessor, class_dist)


# ------------------------- #
# --- EJECUCIÓN DE MAIN --- #
# ------------------------- #
if __name__ == "__main__":
    main()
