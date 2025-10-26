"""
utils.py
--------
Funciones auxiliares para evaluación de modelos y configuración de MLflow.
"""

# ----------------- #
# --- LIBRERÍAS --- #
# ----------------- #
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, matthews_corrcoef, roc_auc_score
)

import mlflow


# ---------------------------- #
# --- FUNCIONES AUXILIARES --- #
# ---------------------------- #
def configure_mlflow(uri: str, experiment_name: str) -> None:
    """Configura el entorno de seguimiento de MLflow."""
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)


def evaluate_model(y_true, y_pred, y_proba=None, average="macro"):
    """Calcula métricas de clasificación con manejo robusto de errores."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average=average, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except ValueError:
            metrics["log_loss"] = None

        try:
            metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
        except ValueError:
            metrics["roc_auc_ovr"] = None

    return metrics
