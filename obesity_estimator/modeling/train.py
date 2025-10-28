"""
train_model.py
---------------
Entrena y ajusta múltiples modelos supervisados, registrando métricas y parámetros en MLflow.
"""


# ----------------- #
# --- LIBRERÍAS --- #
# ----------------- #
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from obesity_estimator.config import (
    RANDOM_STATE, TRAIN_FILEPATH, MODELS_DIR,
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME
)
from obesity_estimator.utils import configure_mlflow

import joblib
import logging
import mlflow
import os
import pandas as pd


# ------------------------------- #
# --- CONFIGURACIÓN Y LOGGING --- #
# ------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------- #
# --- FUNCIONES AUXILIARES --- #
# ---------------------------- #
def train_models(X_train, y_train, scoring="f1_macro"):
    """Entrena y ajusta varios modelos, devolviendo los mejores por tipo."""
    models = {
        "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(max_iter=500, solver="saga", random_state=RANDOM_STATE),
        "SVC": SVC(probability=True, random_state=RANDOM_STATE)
    }

    param_grids = {
        "RandomForest": {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
        "GradientBoosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        "LogisticRegression": {"C": [0.01, 0.1, 1], "penalty": ["l2"]},
        "SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
    }

    best_models = {}

    for name, model in models.items():
        logger.info(f"Entrenando modelo: {name}")
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring=scoring, n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)

        best_models[name] = grid.best_estimator_
        logger.info(f"Mejor {scoring} para {name}: {grid.best_score_:.4f}")

        # Log en MLflow
        with mlflow.start_run(run_name=f"{name}_train"):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric(f"best_cv_{scoring}", grid.best_score_)
            signature = infer_signature(X_train, grid.best_estimator_.predict(X_train))
            mlflow.sklearn.log_model(grid.best_estimator_, name=f"{name}_model", signature=signature)

    return best_models


# ------------------------- #
# --- PROCESO PRINCIPAL --- #
# ------------------------- #
def main():
    """Entrena modelos y guarda los resultados."""
    configure_mlflow(MLFLOW_TRACKING_URI, EXPERIMENT_NAME)

    logger.info("Cargando dataset de entrenamiento...")
    train_df = pd.read_csv(TRAIN_FILEPATH)
    X_train = train_df.drop(columns=["NObeyesdad"])
    y_train = train_df["NObeyesdad"]

    best_models = train_models(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in best_models.items():
        path = os.path.join(MODELS_DIR, f"{name}_best.pkl")
        joblib.dump(model, path)
        logger.info(f"Modelo guardado: {path}")

    logger.info("Entrenamiento finalizado exitosamente.")


# ------------------------- #
# --- EJECUCIÓN DE MAIN --- #
# ------------------------- #
if __name__ == "__main__":
    main()
