#!/bin/zsh
# ==========================================================
# Script: ipynb-to-py.sh
# Descripción: Convierte todos los notebooks .ipynb a .py
#              para que puedan ser ejecutados por DVC.
# ==========================================================

# Define los directorios de notebooks y scripts
NOTEBOOK_DIR="notebooks/a01313663"
SCRIPTS_DIR="src/scripts/a01313663"
CONVERTED=0

echo "Buscando notebooks nuevos o modificados..."

# Recorre todos los notebooks .ipynb en el directorio
for nb in "$NOTEBOOK_DIR"/*.ipynb; do

    # Nombre base del archivo
    base_name=$(basename "$nb" .ipynb)
    py_file="$SCRIPTS_DIR/${base_name}.py"

    # Si el .py no existe o el .ipynb es más reciente, convertir
    if [[ ! -f "$py_file" || "$nb" -nt "$py_file" ]]; then
        echo "Exportando: $nb → $py_file"
        jupyter nbconvert --to script "$nb" --output-dir "$SCRIPTS_DIR" >/dev/null 2>&1
        ((CONVERTED++))
    fi
done

if [[ $CONVERTED -eq 0 ]]; then
    echo "Todos los notebooks están actualizados."
else
    echo "$CONVERTED notebooks convertidos."
fi
