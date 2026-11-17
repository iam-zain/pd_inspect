"""Sensitivity of each candidate on PD patients from the PD-only recruitment waves (PATNO >= 10000).
For candidates trained on the original cohort these 278 patients (220 train-file + 58 test-file) were never used."""
import os, sys, json, numpy as np, pandas as pd
from joblib import load
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE); sys.path.insert(0, BASE)
import min16_lib as L
from run_candidates import load as ld, FEATURES, CFG, NEW
rows = []
for c in ["A_visit_all", "B_patientvisit_all", "C_visit_orig", "D_patientvisit_orig", "E_none_orig"]:
    B = load(os.path.join(HERE, "saved_models", c, "pipeline.joblib"))
    for src in ["train_file", "test_file"]:
        d, X, y, g = ld(CFG[src]); keep = g >= NEW
        p = L.predict_pipeline(B, X[keep].reset_index(drop=True), g[keep])["Final"]
        rows.append({"Candidate": c, "Source": src.replace("_file", ""), "n_patients": len(p), "used_in_training": src == "train_file" and c.endswith("_all"),
                     "mean_P": round(p.mean(), 3), "sensitivity": round((p >= B["threshold"]).mean(), 3)})
df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, "pdonly_waves_sensitivity.csv"), index=False); print(df.to_string(index=False))
