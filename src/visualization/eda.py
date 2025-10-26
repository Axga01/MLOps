"""
eda.py
-----------------
Realiza la operación de EDA sobre el dataset limpio.
"""


# ----------------- #
# --- LIBRERÍAS --- #
# ----------------- #
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

from src.config import PROCESSED_FILEPATH, FIGURES_DIR, NUMERIC_COLS, CATEGORICAL_COLS, BINARY_COLS, TARGET_COL


# ------------------------------- #
# --- CONFIGURACIÓN Y LOGGING --- #
# ------------------------------- #
sns.set_theme(style="whitegrid", palette="muted", context="notebook")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")


# ---------------------------- #
# --- FUNCIONES AUXILIARES --- #
# ---------------------------- #
# Se asegura que el directorio de figuras exista
def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ----------------------------------------- #
# --- FUNCIONES DE ANÁLISIS DESCRIPTIVO --- #
# ----------------------------------------- #
def save_descriptive_stats(df):
    df[NUMERIC_COLS].describe().to_csv(os.path.join(FIGURES_DIR, "numeric_summary.csv"))
    df[CATEGORICAL_COLS].describe().to_csv(os.path.join(FIGURES_DIR, "categorical_summary.csv"))
    df[BINARY_COLS].astype("object").describe().to_csv(os.path.join(FIGURES_DIR, "binary_summary.csv"))
    print(f"Estadísticas descriptivas guardadas en {FIGURES_DIR}")


# ---------------------------------- #
# --- FUNCIONES DE VISUALIZACIÓN --- #
# ---------------------------------- #
# Gráficos para columnas númericas
def plot_numeric_distributions(df):
    for col in NUMERIC_COLS:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.histplot(df[col], kde=True, bins=20, ax=axes[0])
        axes[0].set_title(f"Histograma de {col}")
        sns.boxplot(x=df[col], ax=axes[1], width=0.3)
        axes[1].set_title(f"Boxplot de {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"dist_{col}.png"))
        plt.close(fig)


# Gráficos para columnas categóricas
def plot_categorical_counts(df):
    for col in CATEGORICAL_COLS:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(y=df[col], order=df[col].value_counts().index, palette="viridis", ax=ax, hue=df[col])
        ax.set_title(f"Conteo: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"cat_count_{col}.png"))
        plt.close(fig)


# Gráficos para columnas binarias
def plot_binary_correlations(df):
    fig, axes = plt.subplots(1, len(BINARY_COLS), figsize=(5 * len(BINARY_COLS), 5))
    total = len(df)
    for ax, col in zip(axes, BINARY_COLS):
        sns.countplot(x=df[col], ax=ax, palette="Set2", order=[0, 1], hue=df[col], legend=False)
        ax.set_title(f"{col}")
        ax.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "binary_counts.png"))
    plt.close(fig)

    # Boxplots numéricos vs binarias
    for num_col in NUMERIC_COLS:
        fig, axes = plt.subplots(1, len(BINARY_COLS), figsize=(5 * len(BINARY_COLS), 5), sharey=True)
        for i, bin_col in enumerate(BINARY_COLS):
            sns.boxplot(x=df[bin_col], y=df[num_col], ax=axes[i], width=0.5)
            axes[i].set_title(f"{num_col} según {bin_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"num_vs_bin_{num_col}.png"))
        plt.close(fig)


# Gráficos de relación de la variable objetivo
def plot_target_relationships(df):
    ordered_classes = [
        "insufficient_weight", "normal_weight", "overweight_level_i",
        "overweight_level_ii", "obesity_type_i", "obesity_type_ii", "obesity_type_iii"
    ]
    cat_type = CategoricalDtype(categories=ordered_classes, ordered=True)
    df[TARGET_COL] = df[TARGET_COL].astype(cat_type)

    # Boxplots numéricos vs target
    n_cols = 3
    n_rows = (len(NUMERIC_COLS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        sns.boxplot(x=TARGET_COL, y=col, data=df, ax=axes[i])
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].set_title(f"{col} por categoría de {TARGET_COL}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "num_vs_target.png"))
    plt.close(fig)

    # Barras apiladas categóricas vs target
    object_cols_no_target = [col for col in CATEGORICAL_COLS if col != TARGET_COL]
    for col in object_cols_no_target:
        ct = pd.crosstab(df[col], df[TARGET_COL])
        ct = ct[ordered_classes]
        ct_perc = ct.div(ct.sum(axis=1), axis=0) * 100
        ax = ct_perc.plot(kind="bar", stacked=True, figsize=(8, 4))
        plt.title(f"{col} vs {TARGET_COL} (%)")
        plt.ylabel("Porcentaje")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"cat_vs_target_{col}.png"))
        plt.close()


# ------------------------- #
# --- PROCESO PRINCIPAL --- #
# ------------------------- #
def main():
    print("=== Exploratory Data Analysis ===")
    ensure_figures_dir()
    df = pd.read_csv(PROCESSED_FILEPATH)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    save_descriptive_stats(df)
    plot_numeric_distributions(df)
    plot_categorical_counts(df)
    plot_binary_correlations(df)
    plot_target_relationships(df)

    print(f"\nAnálisis completado. Figuras guardadas en: {FIGURES_DIR}")


# ------------------------- #
# --- EJECUCIÓN DE MAIN --- #
# ------------------------- #
if __name__ == "__main__":
    main()
