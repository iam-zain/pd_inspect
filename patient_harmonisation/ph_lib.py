"""Patient-level missingness harmonisation for the Min16 pipeline (experimental).

Problem being fixed
-------------------
`min16_lib.harmonize_missingness` equalises the *visit-level* recording rate of every
feature between PD and HC. It cannot equalise the *patient-level* pattern: in the
PD-only recruitment waves some items (SCOPA-AUT thermoregulatory/urinary/cardiovascular)
were never recorded at any visit, so after visit-level blanking ~41 % of PD patients but
~7 % of HC patients still have the item missing at every visit. M3 (patient profile) and
the M1 stacker (patient-averaged per-feature probabilities) then see the exact class-
balanced fill value far more often in PD and learn "never recorded => PD".

Fix
---
`harmonize_patient_level` first equalises, per feature, the share of patients in whom the
feature is never recorded, by removing the feature from *all* visits of randomly chosen
patients of the class that records it more often (row floor of MIN_ANSWERED answered items
respected). The existing visit-level harmonisation is then applied to the result, so both
the patient-level and the visit-level recording rates are equal between classes.

Everything else (models, CV, cross-fitting, threshold, metrics) is reused unchanged from
min16_lib, so differences between pipelines come only from the harmonisation step.
"""
import os, sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import min16_lib as L
from min16_lib import (harmonize_missingness, fit_m1_feature, m1_feature_probs, fit_m2, predict_m2,
                       fit_m3, predict_m3, fit_xgb_stacker, predict_xgb_stacker, fit_lr_on_logits,
                       predict_lr_on_logits, crossfit, unit_mean, RANDOM_STATE, N_SPLITS, N_REPEATS,
                       MIN_ANSWERED)


def harmonize_patient_level(X, y, g, features, seed, min_answered=MIN_ANSWERED, then_visit=True, report=False):
    """Equalise the per-class share of patients with each feature never recorded, then
    (optionally) apply the visit-level harmonisation. Training rows only."""
    X = X.copy(); y = np.asarray(y); g = np.asarray(g); rng = np.random.RandomState(seed + 7919)
    pat_y = pd.Series(y).groupby(g).first()
    pats = {k: pat_y.index[pat_y.values == k].values for k in (0, 1)}
    rows_of = pd.Series(np.arange(len(g))).groupby(g).apply(lambda s: s.values).to_dict()
    n_ans = X[features].notna().sum(axis=1).values
    shortfall = {}
    for f in features:
        col = X.columns.get_loc(f)
        obs = X[f].notna().values
        rec = {p: obs[rows_of[p]].any() for k in (0, 1) for p in pats[k]}
        never = {k: np.mean([not rec[p] for p in pats[k]]) for k in (0, 1)}
        target = max(never[0], never[1])
        for k in (0, 1):
            n_need = int(round(target * len(pats[k]) - never[k] * len(pats[k])))
            if n_need <= 0:
                continue
            # eligible: feature recorded at least once, and every visit that records it keeps >= floor
            elig = [p for p in pats[k] if rec[p]
                    and (n_ans[rows_of[p][obs[rows_of[p]]]] > min_answered).all()]
            take = rng.choice(len(elig), min(n_need, len(elig)), replace=False) if elig else []
            for t in take:
                r = rows_of[elig[t]]; r = r[obs[r]]
                X.iloc[r, col] = np.nan; n_ans[r] -= 1
            if len(elig) < n_need:
                shortfall[f] = shortfall.get(f, 0) + n_need - len(elig)
    if then_visit:
        X = harmonize_missingness(X, y, features, seed, min_answered)
    return (X, shortfall) if report else X


def harmonize(X, y, g, features, seed, mode):
    if mode == "visit":
        return harmonize_missingness(X, y, features, seed)
    if mode == "patient+visit":
        # two alternating rounds: one round already equalises the SCOPA-AUT items; the second
        # reduces the gaps the visit-level step re-opens for rarely recorded items (max 3.9 pp)
        for r in range(2):
            X = harmonize_patient_level(X, y, g, features, seed + 1000 * r, then_visit=True)
        return X
    if mode == "patient":
        return harmonize_patient_level(X, y, g, features, seed, then_visit=False)
    if mode == "none":
        return X
    raise ValueError(mode)


def fold_task(X, y, g, tr_i, va_i, features, seed, mode):
    Xtr = harmonize(X.iloc[tr_i], y[tr_i], g[tr_i], features, seed, mode)
    Xva = X.iloc[va_i]
    m1 = {f: fit_m1_feature(Xtr[[f, "age", "sex"]], y[tr_i]) for f in features}
    return {"va_i": va_i, "m1": m1_feature_probs(m1, Xva),
            "m2": predict_m2(fit_m2(Xtr, y[tr_i]), Xva),
            "m3": predict_m3(fit_m3(Xtr, y[tr_i], g[tr_i]), Xva, g[va_i])["M3"]}


def run_oof(X, y, g, features, mode, n_jobs=-1):
    splits = [(a, b) for r in range(N_REPEATS)
              for a, b in StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE + r).split(X, y, g)]
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fold_task)(X, y, g, a, b, features, RANDOM_STATE + k, mode) for k, (a, b) in enumerate(splits))
    n = len(X)
    s1, s2, cnt = {f: np.zeros(n) for f in features}, np.zeros(n), np.zeros(n)
    m3_parts = []
    for r in res:
        for f in features:
            s1[f][r["va_i"]] += r["m1"][f]
        s2[r["va_i"]] += r["m2"]; cnt[r["va_i"]] += 1; m3_parts.append(r["m3"])
    feat_p = unit_mean(g, {f: s1[f] / cnt for f in features})[features]
    oof = pd.DataFrame(index=feat_p.index)
    oof["M2"] = unit_mean(g, {"p": s2 / cnt})["p"]
    oof["M3"] = pd.concat(m3_parts).groupby(level=0).mean()
    oof["y"] = pd.Series(y).groupby(g).first()
    return feat_p, oof


def train_final(X, y, g, feat_p, oof, features, config, mode, n_jobs=-1):
    yp = oof["y"].values
    X = harmonize(X, y, g, features, RANDOM_STATE, mode)
    fitted = Parallel(n_jobs=n_jobs, backend="loky")(delayed(fit_m1_feature)(X[[f, "age", "sex"]], y) for f in features)
    B = {"m1": dict(zip(features, fitted)), "stacker": fit_xgb_stacker(feat_p.values, yp),
         "m2": fit_m2(X, y), "m3": fit_m3(X, y, g)}
    oof["M1"] = crossfit(fit_xgb_stacker, predict_xgb_stacker, feat_p.values, yp, seed_offset=100)
    B["meta"] = fit_lr_on_logits(oof[["M1", "M2", "M3"]].values, yp)
    oof["Final"] = crossfit(fit_lr_on_logits, predict_lr_on_logits, oof[["M1", "M2", "M3"]].values, yp, seed_offset=200)
    fpr, tpr, thr = roc_curve(yp, oof["Final"])
    B["threshold"] = float(thr[np.argmax(tpr - fpr)])
    B["features"], B["config"] = list(features), {**config, "harmonisation": mode}
    return B
