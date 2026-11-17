"""Score new participants (for example the PPMI prodromal cohort) with the final PD-INSPECT model.

Usage
-----
    python score_prodromal.py  prodromal_visits.csv  pipeline.joblib  results_prefix

Input CSV (one row per visit), the same columns as the training matrices:
    PATNO, EVENT_ID          participant and visit identifiers (EVENT_ID optional)
    Age                      age in years at that visit
    Sex                      1 = male, 2 = female (PPMI coding)
    the 30 clinical items    named exactly as in config.json ("features"), left BLANK when not recorded
Extra columns (for example a clinical severity score or SAA result) are ignored.

Do NOT impute, harmonise or rescale anything first: the model fills blanks and applies its own
scaling. A visit is scored only if more than 15 of the 30 items are answered.

Output
------
    <prefix>_per_visit.csv        every visit scored on its own
    <prefix>_per_participant.csv  probabilities averaged over all scored visits of each participant
Columns: M1, M2, M3 (sub-model probabilities), P_PD (final ensemble probability), prediction
(PD if P_PD >= the frozen threshold, otherwise Healthy), n_answered, status.

The pipeline file is NOT in the public repository because it contains PPMI-derived training
profiles (M3 is collaborative filtering). Get it from the study team and keep it private.
Requires min16_lib.py from this repository in the same folder, and the package versions in
requirements.txt (scikit-learn, xgboost, pandas, numpy, joblib).
"""
import os
import sys

import numpy as np
import pandas as pd
from joblib import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from min16_lib import predict_pipeline  # noqa: E402


def score(df, B, per="visit"):
    cfg = B["config"]
    clin, feats = cfg["clinical_columns"], B["features"]
    thr, min_ans = B["threshold"], cfg["min_answered"]
    d = df.copy().reset_index(drop=True)
    for c in clin:
        if c not in d.columns:
            d[c] = np.nan
    d[clin] = d[clin].apply(pd.to_numeric, errors="coerce")
    d["n_answered"] = d[clin].notna().sum(axis=1)
    ok = d["n_answered"] >= min_ans                       # i.e. more than 15 of the 30 items

    X = d[feats].copy()
    X["age"] = pd.to_numeric(d["Age"], errors="coerce")
    X["sex"] = d["Sex"].map({1: 0, 2: 1}).astype(float)

    keep = [c for c in ["PATNO", "EVENT_ID"] if c in d]
    out = d[keep].copy()
    out["n_answered"] = d["n_answered"]
    out["status"] = np.where(ok, "scored", "insufficient data (15 or fewer of 30 items)")
    for c in ["M1", "M2", "M3", "P_PD"]:
        out[c] = np.nan
    if ok.any():
        units = d.loc[ok, "PATNO"].values if per == "participant" else np.where(ok)[0]
        pr = predict_pipeline(B, X[ok], units)
        if per == "participant":
            res = pr.rename(columns={"Final": "P_PD"})[["M1", "M2", "M3", "P_PD"]]
            base = out.groupby("PATNO").agg(n_visits=("status", "size"),
                                            n_visits_scored=("status", lambda s: (s == "scored").sum()))
            out = base.join(res, how="left").reset_index()
            out["status"] = np.where(out["P_PD"].notna(), "scored", "insufficient data")
        else:
            out.loc[ok, ["M1", "M2", "M3", "P_PD"]] = pr[["M1", "M2", "M3", "Final"]].values
    out["prediction"] = np.where(out["P_PD"].isna(), "-", np.where(out["P_PD"] >= thr, "PD", "Healthy"))
    return out


def main(csv_path, pipeline_path, prefix):
    B = load(pipeline_path)
    print(f"Model threshold {B['threshold']:.3f}; features {len(B['features'])}; "
          f"M3 stores {len(B['m3']['z'])} training profiles (K = {B['m3']['k']}, tau = {B['m3']['tau']})")
    df = pd.read_csv(csv_path)
    need = ["PATNO", "Age", "Sex"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"Missing required columns: {miss}")
    present = [c for c in B["config"]["clinical_columns"] if c in df.columns]
    print(f"{len(df)} visits, {df['PATNO'].nunique()} participants; "
          f"{len(present)} of 30 clinical columns found")
    if len(present) < 16:
        sys.exit("Fewer than 16 clinical columns found: check the column names against config.json")
    v = score(df, B, "visit")
    p = score(df, B, "participant")
    v.to_csv(prefix + "_per_visit.csv", index=False)
    p.to_csv(prefix + "_per_participant.csv", index=False)
    print(f"Per visit: {(v.status == 'scored').sum()} scored, "
          f"{(v.prediction == 'PD').sum()} called PD")
    print(f"Per participant: {(p.status == 'scored').sum()} scored, "
          f"{(p.prediction == 'PD').sum()} called PD")
    print("Saved:", prefix + "_per_visit.csv", "and", prefix + "_per_participant.csv")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
