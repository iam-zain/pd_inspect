# Data

No PPMI data are included in this repository. The data are available to qualified
researchers from the Parkinson's Progression Markers Initiative after registration and
approval of a data use agreement: https://www.ppmi-info.org/access-data-specimens/download-data

## Expected input

The notebooks read visit-level sparse matrices (one row per participant visit) with the
columns `PATNO`, `EVENT_ID`, `APPRDX` (1 = PD, 2 = healthy control), `Age`, `Sex`
(1 = male, 2 = female) and the clinical items listed in `config.json`
(`clinical_columns`). Unanswered items are left blank. `Make_Datasets.ipynb` drops
handedness and the MDS-UPDRS items and keeps visits with at least 16 of the 30 model items.

1. Place the source matrices in `data/source/Train/` and `data/source/Test/`
   (`Sparse_Matrix_Min5_*.csv` and `Sparse_Matrix_Min50_*.csv`).
2. Run `Make_Datasets.ipynb`. It writes the analysis files to `Train/` and `Test/`
   in the repository root, which `config.json` points to.

The train/test split is made at the participant level, so no participant appears in both
files. Participants with `PATNO >= 10000` belong to later PD-only PPMI recruitment
waves; they are excluded from the primary training data and used only as held-out checks
(`new_cohort_min_patno` in `config.json`).
