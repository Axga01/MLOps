"""
evaluate_model.py
-----------------
Evalúa los modelos entrenados sobre el conjunto de prueba, genera métricas y matriz de confusión.
"""


# ----------------- #
# --- LIBRERÍAS --- #
# ----------------- #
from obesity_estimator.config import TEST_FILEPATH, MODELS_DIR, REPORTS_DIR
from obesity_estimator.utils import evaluate_model

import joblib
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ------------------------------- #
# --- CONFIGURACIÓN Y LOGGING --- #
# ------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ------------------------- #
# --- PROCESO PRINCIPAL --- #
# ------------------------- #
def main():
    logger.info("Cargando dataset de prueba...")
    test_df = pd.read_csv(TEST_FILEPATH)
    X_test = test_df.drop(columns=["NObeyesdad"])
    y_test = test_df["NObeyesdad"]

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_best.pkl")]
    results = []

    for file in model_files:
        name = file.replace("_best.pkl", "")
        logger.info(f"Evaluando modelo: {name}")
        model = joblib.load(os.path.join(MODELS_DIR, file))

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        metrics = evaluate_model(y_test, y_pred, y_proba)
        results.append({"Modelo": name, **metrics})

    # Comparación de resultados
    results_df = pd.DataFrame(results).sort_values(by="f1_macro", ascending=False)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    results_path = os.path.join(REPORTS_DIR, "evaluation_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Métricas guardadas en: {results_path}")

    best_model_name = results_df.iloc[0]["Modelo"]
    logger.info(f"Mejor modelo: {best_model_name}")

    # Matriz de confusión
    best_model = joblib.load(os.path.join(MODELS_DIR, f"{best_model_name}_best.pkl"))
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    cm_df = pd.DataFrame(cm, index=best_model.classes_, columns=best_model.classes_)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusión - {best_model_name}")
    plt.ylabel("Real")
    plt.xlabel("Predicho")

    confusion_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(confusion_path)
    plt.close()

    cm_df.to_csv(os.path.join(REPORTS_DIR, "confusion_matrix.csv"))
    logger.info(f"Matriz de confusión guardada en: {confusion_path}")

    logger.info("Evaluación finalizada exitosamente.")


# ------------------------- #
# --- EJECUCIÓN DE MAIN --- #
# ------------------------- #
if __name__ == "__main__":
    main()
