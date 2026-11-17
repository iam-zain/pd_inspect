"""
Builds saved_models/sample_bank.json - the real, de-identified PPMI records the
web tool's "Random 16 / Random 30" quick-fill buttons draw from.

Every record is ONE real visit row (not a cross-visit average), so every value is a
literal answer a real participant gave - nothing here is generated, interpolated or
perturbed. Only records with Smell + Gastrointestinal recorded and >=16 total
answered features are included (the same MIN_ANSWERED inclusion rule the model
itself was trained under). No PATNO or other identifier is kept - only the clinical
scores, age and sex.

Run this after retraining (or whenever Train/Test CSVs change), then copy the output
into pythonanywhere_upload/saved_models/sample_bank.json before redeploying.
"""
import json
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_ANSWERED = 16


def build(config_path=os.path.join(BASE_DIR, "config.json"),
          out_path=os.path.join(BASE_DIR, "saved_models", "sample_bank.json")):
    with open(config_path) as fh:
        cfg = json.load(fh)
    features = cfg["features"]

    d = pd.concat([pd.read_csv(cfg["train_file"]), pd.read_csv(cfg["test_file"])], ignore_index=True)
    X = pd.DataFrame({f: pd.to_numeric(d[f], errors="coerce") for f in features})
    X["age"] = d["Age"].astype(float)
    X["sex"] = d["Sex"].map({1: 0, 2: 1}).astype(float)          # 0 = male, 1 = female
    X["y"] = (d["APPRDX"] == 1).astype(int).values               # 1 = PD, 0 = healthy

    bank = {"healthy": [], "pd": []}
    for _, row in X.iterrows():
        feat_vals = row[features].dropna()
        if len(feat_vals) < MIN_ANSWERED:
            continue
        if "Smell" not in feat_vals.index or "Gastrointestinal" not in feat_vals.index:
            continue
        if pd.isna(row["age"]) or pd.isna(row["sex"]):
            continue
        rec = {
            "features": {k: (int(v) if float(v).is_integer() else round(float(v), 2))
                        for k, v in feat_vals.items()},
            "age": round(float(row["age"]), 1),
            "sex": float(row["sex"]),
        }
        bank["healthy" if row["y"] == 0 else "pd"].append(rec)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(bank, fh, separators=(",", ":"))
    print(f"wrote {out_path}: {len(bank['healthy'])} healthy, {len(bank['pd'])} PD real records "
          f"({os.path.getsize(out_path) / 1024:.0f} KB)")
    return bank


if __name__ == "__main__":
    build()
