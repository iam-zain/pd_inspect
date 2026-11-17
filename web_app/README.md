# PD-INSPECT web application

Flask application behind https://pdinspect.pythonanywhere.com/. It accepts age, sex and the
30 clinical items, requires at least 16 answered items, and returns the final ensemble
probability, the votes of M1, M2 and M3, a confidence band and a downloadable PDF report.

## Run locally

```bash
cd web_app
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000/. `app.py` starts Flask's development server in debug mode,
which is meant for local use only. On a public server, run the app through a WSGI server
(for example the PythonAnywhere web app configuration).

## Files

| Path | Purpose |
|---|---|
| `app.py` | Routes: form page, `/submit`, PDF report, `/sample_patient` |
| `prediction.py` | Input checks and prediction with the saved pipeline |
| `min16_lib.py` | Same library as in the repository root (needed to load the pipeline) |
| `pdf_report.py` | PDF report generation (reportlab) |
| `saved_models/pipeline.joblib` | Trained final pipeline - **not included** (see below) |
| `saved_models/config.json` | Feature list, threshold and settings |
| `static/`, `templates/` | Front end |

xgboost is not needed at serving time: the trained boosters are stored as NumPy arrays.

## Quick-fill examples

The "Random 16 / Random 30" buttons draw example records from
`saved_models/sample_bank.json`. That file contains real PPMI visit records, so it is
**not** included here. Without it the app works normally, and only those two buttons
return "Sample bank not available". If you have PPMI access, build your own copy with
`build_sample_bank.py` in the repository root and place it in `saved_models/`.

## Trained model file

`saved_models/pipeline.joblib` is not included in this repository. M3 is a collaborative-filtering
model, so the trained pipeline stores the harmonised profiles of the PPMI training participants it
compares new inputs with. These are derived from PPMI data and cannot be shared publicly. With PPMI
access, run the notebooks (`Train_3SubModels.ipynb` saves `saved_models/primary/pipeline.joblib`) and
convert it for serving with `min16_lib.strip_xgboost`.
