"""Train and compare harmonisation candidates. Writes only inside patient_harmonisation/.
Run with the Anaconda Python 3.13 environment:  python run_candidates.py [candidate ...]"""
import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, BASE)
import ph_lib as PH
import min16_lib as L

CFG = json.load(open(os.path.join(BASE, "config.json")))
FEATURES, CLIN30, MIN_ANS = CFG["features"], CFG["clinical_columns"], CFG["min_answered"]
NEW = CFG["new_cohort_min_patno"]
CANDS = {"A_visit_all": ("visit", None), "B_patientvisit_all": ("patient+visit", None),
         "C_visit_orig": ("visit", NEW), "D_patientvisit_orig": ("patient+visit", NEW), "E_none_orig": ("none", NEW)}

def load(path, max_patno=None):
    d = pd.read_csv(path)
    if max_patno is not None: d = d[d.PATNO < max_patno]
    d = d.reset_index(drop=True)
    X = pd.DataFrame({f: pd.to_numeric(d[f], errors="coerce") for f in FEATURES})
    X["age"] = d["Age"].astype(float); X["sex"] = d["Sex"].map({1: 0, 2: 1}).astype(float)
    return d, X[FEATURES + ["age", "sex"]], (d["APPRDX"] == 1).astype(int).values, d["PATNO"].values

def negative_controls(X, y, g):
    share = X[FEATURES].notna().groupby(g).mean()
    yp = pd.Series(y).groupby(g).first().loc[share.index].values
    never = (share == 0).astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=42); spw = (yp == 0).sum() / (yp == 1).sum()
    xgb = lambda: XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.05, scale_pos_weight=spw, random_state=42, n_jobs=4)
    out = {}
    for nm, M, A in [("share_LR", LogisticRegression(max_iter=5000, class_weight="balanced"), share.values),
                     ("share_XGB", xgb(), share.values), ("never_XGB", xgb(), never.values)]:
        out[nm] = roc_auc_score(yp, cross_val_predict(M, A, yp, cv=cv, method="predict_proba")[:, 1])
    out["never_thermo_PD"], out["never_thermo_HC"] = never.Thermoregulatory[yp == 1].mean(), never.Thermoregulatory[yp == 0].mean()
    return out

def score_rows(B, d):
    """Single-visit scoring of raw rows (as the web tool does); rows with < MIN_ANS of 30 answered are dropped."""
    ok = d[CLIN30].notna().sum(axis=1) >= MIN_ANS
    d = d[ok]
    X = d[FEATURES].apply(pd.to_numeric, errors="coerce").copy()
    X["age"] = d["Age"].astype(float); X["sex"] = d["Sex"].map({1: 0, 2: 1}).astype(float)
    p = L.predict_pipeline(B, X.reset_index(drop=True), np.arange(len(X)))["Final"].values
    return p

def stress(B, te):
    rng = np.random.RandomState(11); rows = []
    coh = te.PATNO >= NEW
    masks = te[coh][CLIN30].notna().values
    for grp, base in [("HC", te[te.APPRDX == 2]), ("PD_orig", te[(te.APPRDX == 1) & ~coh])]:
        m = masks[rng.randint(0, len(masks), len(base))]
        cm = base.copy(); cm[CLIN30] = base[CLIN30].where(m)
        keep = cm[CLIN30].notna().sum(axis=1) >= MIN_ANS; cm, b0 = cm[keep], base[keep]
        rc = b0.copy()
        for i in rc.index:
            ans = [c for c in CLIN30 if pd.notna(b0.at[i, c])]
            k = len(ans) - cm.loc[i, CLIN30].notna().sum()
            if k > 0: rc.loc[i, list(rng.choice(ans, k, replace=False))] = np.nan
        rb = b0.copy()
        for i in rb.index:
            ans = [c for c in CLIN30 if pd.notna(b0.at[i, c])]; nd = max(0, len(ans) - MIN_ANS)
            if nd: rb.loc[i, list(rng.choice(ans, rng.randint(0, nd + 1), replace=False))] = np.nan
        sc = b0.copy(); sc[["Thermoregulatory", "Urinary_SCOPA", "Cardiovascular"]] = np.nan
        sc = sc[sc[CLIN30].notna().sum(axis=1) >= MIN_ANS]
        for nm, dd in [("orig", b0), ("random_extra", rb), ("matched_random", rc), ("cohort_mask", cm), ("scopa3_removed", sc)]:
            p = score_rows(B, dd)
            rows.append({"group": grp, "cond": nm, "n": len(p), "meanP": p.mean(), "calledPD": (p >= B["threshold"]).mean()})
    X0 = pd.DataFrame([{f: np.nan for f in FEATURES}]); X0["age"] = 64.0; X0["sex"] = 0.0
    allblank = L.predict_pipeline(B, X0, np.array([0])).iloc[0].to_dict()
    return pd.DataFrame(rows), allblank

def run(name):
    mode, max_patno = CANDS[name]
    t0 = time.time(); rdir = os.path.join(HERE, "results", name); mdir = os.path.join(HERE, "saved_models", name)
    os.makedirs(rdir, exist_ok=True); os.makedirs(mdir, exist_ok=True)
    d, X, y, g = load(CFG["train_file"], max_patno)
    te_full, Xt, yt, gt = load(CFG["test_file"], None)
    print(f"[{name}] mode={mode} train {len(np.unique(g))} patients / {len(X)} visits", flush=True)
    feat_p, oof = PH.run_oof(X, y, g, FEATURES, mode)
    B = PH.train_final(X, y, g, feat_p, oof, FEATURES, CFG, mode)
    dump(B, os.path.join(mdir, "pipeline.joblib"))
    pred = L.predict_pipeline(B, Xt, gt); ypt = pd.Series(yt).groupby(gt).first().loc[pred.index].values
    pred_v = L.predict_pipeline(B, Xt, np.arange(len(Xt)))
    pred.assign(y=ypt).to_csv(os.path.join(rdir, "test_predictions_patient.csv"))
    thr = B["threshold"]; o = oof.dropna(); orig = pred.index.values < NEW
    m = pd.concat([L.evaluate(o, o["y"].values, o.index.values, thr, "train OOF (patient)"),
                   L.evaluate(pred, ypt, pred.index.values, thr, "TEST patient, all"),
                   L.evaluate(pred[orig], ypt[orig], pred.index.values[orig], thr, "TEST patient, original cohort"),
                   L.evaluate(pred_v, yt, gt, thr, "TEST single visit, all")])
    m.insert(0, "Candidate", name); m.to_csv(os.path.join(rdir, "metrics.csv"), index=False)
    comp = []
    for sub, keep in [("all", np.ones(len(pred), bool)), ("original", orig)]:
        for c in ["M1", "M2", "M3"]:
            dl, lo, hi, pv = L.auc_diff_ci(ypt[keep], pred["Final"].values[keep], pred[c].values[keep], pred.index.values[keep])
            comp.append({"Candidate": name, "Test": sub, "Comparison": f"Final - {c}", "dAUC": dl, "lo": lo, "hi": hi, "p": pv})
    pd.DataFrame(comp).to_csv(os.path.join(rdir, "auc_comparison.csv"), index=False)
    Xh = PH.harmonize(X, y, g, FEATURES, L.RANDOM_STATE, mode)
    nc = negative_controls(Xh, y, g); nc["observed_values_removed_pct"] = 100 * (1 - Xh[FEATURES].notna().values.sum() / X[FEATURES].notna().values.sum())
    st, allblank = stress(B, te_full)
    st.insert(0, "Candidate", name); st.to_csv(os.path.join(rdir, "stress_tests.csv"), index=False)
    info = {"candidate": name, "mode": mode, "max_patno": max_patno, "threshold": thr,
            "meta_coef": B["meta"].coef_[0].tolist(), "meta_intercept": float(B["meta"].intercept_[0]),
            "allblank": allblank, "negative_controls": nc, "minutes": (time.time() - t0) / 60}
    json.dump(info, open(os.path.join(rdir, "info.json"), "w"), indent=2, default=float)
    print(f"[{name}] done in {info['minutes']:.1f} min; thr {thr:.3f}; negctl {nc}", flush=True)

if __name__ == "__main__":
    for nm in (sys.argv[1:] or list(CANDS)):
        run(nm)
