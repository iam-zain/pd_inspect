# PD-INSPECT

Code for **"Interpretable Symptom-Based Machine Learning for Parkinson's Disease Prediction: A Feasibility Study"**.

PD-INSPECT estimates the probability of Parkinson's disease (PD) from 30 non-motor and
cognitive items plus age and sex, using data from the Parkinson's Progression Markers
Initiative (PPMI). It is a stacked ensemble of three sub-models, built with explicit
controls for missingness-related data leakage.

- **Web tool:** https://pdinspect.pythonanywhere.com/
- **Data:** not included; see [data/README.md](data/README.md)

## Model

| Sub-model | Name | What it does |
|---|---|---|
| M1 | Symptom-wise ensemble | Logistic regression, calibrated linear SVM and XGBoost per item, combined by a class-balanced XGBoost stacker |
| M2 | Joint-symptom model | One XGBoost model on all items of a visit together |
| M3 | Participant-profile model | Collaborative filtering (variant V4): each participant's visit-averaged, standardised profile is matched to training participants by a Euclidean similarity in which every item is weighted by its coefficient in a class-balanced logistic regression; the 80 most similar participants vote with softmax weights (temperature 0.05) and class-balanced vote weights |
| Final | Meta-learner | L2 logistic regression on the cross-fitted out-of-fold logits of M1, M2 and M3 |

Each sub-model was chosen from 12 variants. All cross-validation is grouped by participant
(StratifiedGroupKFold, 5 folds × 3 repeats), every stacking layer is cross-fitted, and the
decision threshold (Youden J, 0.611) is fixed on training data before the test set is used.

**Leakage control.** Missingness is harmonised between PD and healthy controls inside every
training fold at two levels: the share of participants in whom an item is never recorded,
and the per-visit recording rate. Imputation is class-balanced. Missingness-only negative
controls and stress tests check that blank entries carry no label information.

**Held-out test set (123 participants: 83 PD, 40 HC):** AUC 0.940 (95% CI 0.895–0.977),
sensitivity 0.916, specificity 0.775, Brier score 0.092 (lower than each sub-model: M1 0.098, M2 0.118, M3 0.103), calibration slope 0.92.

## Repository layout

| Path | Contents |
|---|---|
| `Make_Datasets.ipynb` | Builds the analysis matrices from the PPMI source matrices |
| `Data_Audit.ipynb` | Item coverage, effect sizes, missingness negative controls |
| `M3_Redesign_Analysis.ipynb` | Evaluation of the M3 collaborative-filtering variants inside the final pipeline, from which the final M3 (V4) was chosen (run after `Train_3SubModels.ipynb`) |
| `Various_M1_MiniModel.ipynb`, `Various_M2_MiniModel.ipynb`, `Various_M3_MiniModel.ipynb` | Exploratory search over 12 variants per sub-model |
| `Train_3SubModels.ipynb` | Cross-fitted training of M1–M3 and the meta-learner; evaluation, calibration, robustness tests; saves the final pipeline |
| `Predict_New_Patients.ipynb` | Applies the saved pipeline to new records (example: `dummy_patients.csv`, synthetic) |
| `SHAP_Top10_AllModels.ipynb` | SHAP explanations for the sub-models and the whole pipeline |
| `min16_lib.py` | Shared library: harmonisation, imputation, sub-models, stacking, metrics |
| `config.json` | Feature list, inclusion rule (≥ 16 answered items) and settings |
| `M3_Redesign_Results/` | The 12 M3 variants (Table 5 of the article) evaluated inside the final pipeline (aggregated results only) |
| `patient_harmonisation/` | Comparison of harmonisation strategies and training cohorts (candidates A–E) |
| `results/` | Aggregated metrics and figures (no per-participant rows) |
| `web_app/` | Flask application behind the web tool |
| `build_sample_bank.py` | Builds the web tool's quick-fill examples from your own PPMI files |
| `run_notebooks.sh` | Re-runs the main notebooks in order |

## Reproducing the analysis

```bash
pip install -r requirements.txt
```

1. Obtain the PPMI data and prepare the input files as described in [data/README.md](data/README.md).
2. Run `Make_Datasets.ipynb`, then `Data_Audit.ipynb`.
3. Optionally run the three `Various_M*_MiniModel.ipynb` notebooks (variant search).
4. Run `Train_3SubModels.ipynb`, `Predict_New_Patients.ipynb` and `SHAP_Top10_AllModels.ipynb`,
   or run `bash run_notebooks.sh` from the repository root.

All notebooks expect to be run from the repository root. Random seeds are fixed (42).
Notebook outputs that displayed participant-level rows were removed before publication.

## Web application

See [web_app/README.md](web_app/README.md).

## Citation

Please cite the article (details to be added on publication).

## Disclaimer

PD-INSPECT is a research tool. It is not a medical device and must not be used for diagnosis.
