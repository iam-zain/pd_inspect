#!/bin/bash
# Re-runs the main pipeline notebooks in order from the repository root.
# Requires the PPMI-derived input matrices in Train/ and Test/ (see data/README.md).
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
for nb in Data_Audit Train_3SubModels Predict_New_Patients SHAP_Top10_AllModels; do
  echo "=== $nb $(date +%T)"
  "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
     --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nb.ipynb"
done
echo "=== all done $(date +%T)"
