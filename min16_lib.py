"""Model, CV and metric code for the Min16 robust pipeline.

Kept in a module (not inside the notebook) so joblib/loky worker processes can
import it on Windows. Imported by 02_Train_Evaluate.ipynb and 03_Predict_New_Patients.ipynb.
Every fit_* function learns all its statistics from the rows it is given, so it is
leakage-free when called inside a CV fold.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
try:
    from xgboost import XGBClassifier
except ImportError:          # prediction-only hosts need no xgboost (see NumpyBooster)
    XGBClassifier = None

RANDOM_STATE = 42
N_SPLITS, N_REPEATS = 5, 3
N_BOOT = 2000
EPS    = 1e-4        # probability clipping before logit
MIN_ANSWERED = 16    # visit-inclusion rule; harmonisation never takes a row below it
DEMO   = ["age", "sex"]


def _spw(y):
    n_pos = (y == 1).sum()
    return float((y == 0).sum() / n_pos) if n_pos else 1.0

def _lg(p):
    return logit(np.clip(p, EPS, 1 - EPS))

def harmonize_missingness(X, y, features, seed, min_answered=MIN_ANSWERED, report=False):
    """
    Missingness harmonisation (TRAINING rows only, never test rows).

    For every feature, the class whose value is recorded more often has randomly chosen
    values blanked until both classes have the same recording rate. The blank/filled
    pattern then carries no label information, so the models cannot learn
    "blank => PD", while every feature is kept. Only observed values are removed;
    nothing is invented.

    Row floor: a row is never taken below `min_answered` answered features, so the
    harmonised training rows still satisfy the study's visit-inclusion rule. Rows that
    are already at the floor are simply not eligible to lose another value. If the floor
    leaves too few eligible rows, a small coverage gap can remain; `report=True` returns
    it alongside the data so it can be checked.
    """
    X = X.copy(); y = np.asarray(y); rng = np.random.RandomState(seed)
    n = {k: (y == k).sum() for k in (0, 1)}
    n_ans = X[features].notna().sum(axis=1).values          # answered count per row, kept up to date
    residual = {}
    for f in features:
        col = X.columns.get_loc(f)
        obs = X[f].notna().values
        target = min(obs[y == 0].mean(), obs[y == 1].mean())
        for k in (0, 1):
            n_drop = int(round(((y == k) & obs).sum() - target * n[k]))
            if n_drop <= 0:
                continue
            free = np.where((y == k) & obs & (n_ans > min_answered))[0]   # row floor respected
            take = rng.choice(free, min(n_drop, len(free)), replace=False)
            X.iloc[take, col] = np.nan
            n_ans[take] -= 1
            if len(free) < n_drop:
                residual[f] = residual.get(f, 0) + n_drop - len(free)
    return (X, residual) if report else X

def harmonize_patient_level(X, y, g, features, seed, min_answered=MIN_ANSWERED, report=False):
    """
    Patient-level missingness harmonisation (TRAINING rows only, never test rows).

    `harmonize_missingness` equalises the visit-level recording rate, but M3 (patient
    profile, collaborative filtering) and the M1 stacker (patient-averaged per-feature probabilities) see whether a
    feature was EVER recorded for a patient. Some PPMI recruitment waves never recorded
    certain items, so "never recorded" stays PD-specific after visit-level blanking.

    Here, for every feature, the share of patients in whom the feature is never recorded
    is equalised between classes by removing the feature from ALL visits of randomly
    chosen patients of the class that records it more often. Only observed values are
    removed, and every affected visit keeps >= `min_answered` answered features.
    """
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
            n_need = int(round((target - never[k]) * len(pats[k])))
            if n_need <= 0:
                continue
            elig = [p for p in pats[k] if rec[p] and (n_ans[rows_of[p][obs[rows_of[p]]]] > min_answered).all()]
            take = rng.choice(len(elig), min(n_need, len(elig)), replace=False) if elig else []
            for t in take:
                r = rows_of[elig[t]]; r = r[obs[r]]
                X.iloc[r, col] = np.nan; n_ans[r] -= 1
            if len(elig) < n_need:
                shortfall[f] = shortfall.get(f, 0) + n_need - len(elig)
    return (X, shortfall) if report else X

# Harmonisation used by the pipeline: "patient+visit" (two alternating rounds of the
# patient-level and the visit-level step; adopted 2026-10-18), or "visit" (earlier version).
HARMONISATION = "patient+visit"

def harmonize(X, y, g, features, seed, mode=None):
    """Apply the pipeline's missingness harmonisation to training rows."""
    mode = mode or HARMONISATION
    if mode == "visit":
        return harmonize_missingness(X, y, features, seed)
    if mode == "patient+visit":
        for r in range(2):
            X = harmonize_patient_level(X, y, g, features, seed + 1000 * r)
            X = harmonize_missingness(X, y, features, seed + 1000 * r)
        return X
    if mode == "none":
        return X
    raise ValueError(f"unknown harmonisation mode {mode!r}")

def balanced_center(X, y, how="median"):
    """
    Class-balanced fill value per column: midpoint of the PD and the HC statistic.
    A plain median of a 78 %-PD training set is itself PD-like, so filling a blank with it
    would push the prediction toward PD. The midpoint is neutral between the classes.
    Falls back to the pooled statistic when one class has no observed value.
    """
    y = np.asarray(y)
    f = (lambda D: D.median()) if how == "median" else (lambda D: D.mean())
    a, b, pooled = f(X[y == 1]), f(X[y == 0]), f(X)
    return ((a + b) / 2).fillna(pooled)

def unit_mean(units, values):
    """Average row-level values within each unit (patient, or visit for single-visit scoring)."""
    return pd.DataFrame(values).groupby(np.asarray(units)).mean()

# ---------- xgboost-free scoring of the fitted boosters ----------
class NumpyBooster:
    """A fitted XGBClassifier flattened into numpy arrays.

    Scoring a gradient-boosted tree is just walking each tree and summing leaves, so
    the trained model can be served without the xgboost package (~300 MB, which does
    not fit the deployment host). `from_xgb` is used once at packaging time; the
    stored object exposes `predict_proba` and is a drop-in for the classifier.
    Agreement with xgboost was verified to 2e-7, i.e. float32 rounding.
    """

    def __init__(self, trees, base_margin):
        self.trees = trees
        self.base_margin = base_margin

    @classmethod
    def from_xgb(cls, clf):
        import json
        bst = clf.get_booster()
        cfg = json.loads(bst.save_config())
        # xgboost >= 2 writes base_score as a vector string, e.g. "[5E-1]"
        raw   = str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]")
        base  = float(raw.split(",")[0])
        trees = []
        for dump in bst.get_dump(dump_format="json"):
            feat, thr, left, right, miss, leaf = [], [], [], [], [], []

            def walk(n):
                i = len(feat)
                feat.append(-1); thr.append(0.0); left.append(-1)
                right.append(-1); miss.append(-1); leaf.append(0.0)
                if "leaf" in n:
                    leaf[i] = float(n["leaf"])
                    return i
                feat[i] = int(str(n["split"]).lstrip("f"))
                thr[i]  = float(n["split_condition"])
                kids    = {c["nodeid"]: c for c in n["children"]}
                left[i]  = walk(kids[n["yes"]])
                right[i] = walk(kids[n["no"]])
                miss[i]  = left[i] if n["missing"] == n["yes"] else right[i]
                return i

            walk(json.loads(dump))
            # thr/A must both be float32: xgboost splits and compares in float32
            # internally, and get_dump() prints split_condition at float32
            # precision. Comparing in float64 turns "v == thr" ties (routine
            # with histogram-picked thresholds) into spurious "v < thr" or
            # "v > thr", misrouting rows into the wrong leaf.
            trees.append((np.asarray(feat, np.int32), np.asarray(thr, np.float32),
                          np.asarray(left, np.int32), np.asarray(right, np.int32),
                          np.asarray(miss, np.int32), np.asarray(leaf, np.float64)))
        return cls(trees, float(np.log(base / (1.0 - base))))

    def _margin(self, A):
        A = np.asarray(A, dtype=np.float32)
        out = np.full(len(A), self.base_margin, dtype=np.float64)
        for feat, thr, left, right, miss, leaf in self.trees:
            node   = np.zeros(len(A), dtype=np.int32)
            active = feat[node] >= 0
            while active.any():
                i = node[active]
                v = A[active, feat[i]]
                node[active] = np.where(np.isnan(v), miss[i],
                                        np.where(v < thr[i], left[i], right[i]))
                active = feat[node] >= 0
            out += leaf[node]
        return out

    def predict_proba(self, A):
        p = expit(self._margin(A))
        return np.column_stack([1.0 - p, p])


def strip_xgboost(B):
    """Return a copy of a fitted bundle whose boosters are NumpyBoosters."""
    B = dict(B)
    B["m1"] = {f: {**b, "xgb": NumpyBooster.from_xgb(b["xgb"])} for f, b in B["m1"].items()}
    B["stacker"] = NumpyBooster.from_xgb(B["stacker"])
    B["m2"] = {**B["m2"], "xgb": NumpyBooster.from_xgb(B["m2"]["xgb"])}
    return B


# ---------- M1: per-feature trio (LR + linear SVM + shallow XGB) ----------
def fit_m1_feature(Xf, y):
    med = balanced_center(Xf, y)
    Xi  = Xf.fillna(med)
    sc  = StandardScaler().fit(Xi)
    Xs  = sc.transform(Xi)
    lr  = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE).fit(Xs, y)
    svm = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
                                 cv=3, method="sigmoid").fit(Xs, y)
    xgb = XGBClassifier(max_depth=2, n_estimators=100, learning_rate=0.1, scale_pos_weight=_spw(y),
                        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1).fit(Xi.values, y)
    return {"medians": med, "scaler": sc, "lr": lr, "svm": svm, "xgb": xgb}

def predict_m1_feature(b, Xf):
    Xi = Xf.fillna(b["medians"])
    Xs = b["scaler"].transform(Xi)
    return np.mean([b["lr"].predict_proba(Xs)[:, 1], b["svm"].predict_proba(Xs)[:, 1],
                    b["xgb"].predict_proba(Xi.values)[:, 1]], axis=0)

def m1_feature_probs(m1, X):
    return {f: predict_m1_feature(b, X[[f, "age", "sex"]]) for f, b in m1.items()}

# ---------- meta: L2 logistic regression on logits ----------
def fit_lr_on_logits(P, y, C=1.0):
    return LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE).fit(_lg(np.asarray(P)), y)

def predict_lr_on_logits(model, P):
    return model.predict_proba(_lg(np.asarray(P)))[:, 1]

# ---------- M1 stacker: XGBoost, class-balanced (mini-model V8) ----------
def fit_xgb_stacker(P, y):
    return XGBClassifier(max_depth=3, n_estimators=150, learning_rate=0.05,
                         scale_pos_weight=_spw(y), eval_metric="logloss",
                         random_state=RANDOM_STATE, n_jobs=1).fit(np.asarray(P), y)

def predict_xgb_stacker(model, P):
    return model.predict_proba(np.asarray(P))[:, 1]

def crossfit(fit_fn, predict_fn, P, y, seed_offset=0):
    """Out-of-fold predictions of a model (patient rows) - used as honest meta input."""
    P, tot, cnt = np.asarray(P), np.zeros(len(y)), np.zeros(len(y))
    for r in range(N_REPEATS):
        for tr_i, va_i in StratifiedKFold(N_SPLITS, shuffle=True,
                                          random_state=RANDOM_STATE + seed_offset + r).split(P, y):
            tot[va_i] += predict_fn(fit_fn(P[tr_i], y[tr_i]), P[va_i])
            cnt[va_i] += 1
    return tot / cnt

# ---------- M2: global XGBoost (params of mini-model V2), imputed inputs ----------
def fit_m2(X, y):
    med = balanced_center(X, y)
    xgb = XGBClassifier(max_depth=4, n_estimators=400, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=_spw(y), eval_metric="logloss",
                        random_state=RANDOM_STATE, n_jobs=1).fit(X.fillna(med).values, y)
    return {"medians": med, "xgb": xgb}

def predict_m2(b, X):
    return b["xgb"].predict_proba(X.fillna(b["medians"]).values)[:, 1]

# ---------- M3: collaborative filtering on patient profiles (variant L10, v6) ----------
# Each participant is summarised as one profile (feature means over visits; features never
# recorded are filled with the class-balanced mean) and standardised. A new profile is compared
# with every training participant by a Euclidean RBF similarity, exp(-distance / sigma), after the
# features are weighted by |coefficient| of an L2 logistic regression fitted on the TRAINING profiles
# only (so it is learned inside every CV fold). The CF_K = 80 most similar participants vote with
# softmax weights exp((sim - max sim) / CF_TAU), each vote re-weighted by the inverse frequency of its
# class (prior-corrected). The output is the weighted PD fraction. Everything is numpy at prediction time.
CF_K, CF_TAU, CF_LR_C = 80, 0.05, 0.1

def fit_m3(X, y, units):
    prof = unit_mean(units, X)                                   # one row per training patient
    yp   = pd.Series(y).groupby(np.asarray(units)).first().loc[prof.index].values.astype(float)
    means = balanced_center(prof, yp, how="mean")
    Xi    = prof.fillna(means).values.astype(float)
    mu, sd = Xi.mean(axis=0), Xi.std(axis=0); sd[sd == 0] = 1.0
    Z = (Xi - mu) / sd
    coef = np.abs(LogisticRegression(C=CF_LR_C, class_weight="balanced", max_iter=2000,
                                     random_state=RANDOM_STATE).fit(Z, yp).coef_[0])
    w  = coef / max(coef.mean(), 1e-12)                          # feature weights, mean 1
    sw = np.sqrt(w); Zw = Z * sw                                 # weighted coordinates
    D = np.sqrt(np.maximum(((Zw[:, None, :] - Zw[None, :, :]) ** 2).sum(axis=2), 0.0))
    sigma = float(np.median(D[np.triu_indices(len(Zw), k=1)])) + 1e-8   # fixed RBF width
    n1 = float(yp.sum()); n0 = float(len(yp) - n1)
    cls_w = np.array([len(yp) / (2 * n0), len(yp) / (2 * n1)])   # inverse class frequency (HC, PD)
    return {"means": means, "mu": mu, "sd": sd, "sw": sw, "feat_w": w, "z": Zw, "y": yp,
            "sigma": sigma, "k": CF_K, "tau": CF_TAU, "cls_w": cls_w}

def predict_m3(b, X, units):
    prof = unit_mean(units, X)
    Xq   = prof.fillna(b["means"]).values.astype(float)
    Zq   = ((Xq - b["mu"]) / b["sd"]) * b["sw"]
    D    = np.sqrt(np.maximum(((Zq[:, None, :] - b["z"][None, :, :]) ** 2).sum(axis=2), 0.0))
    S    = np.exp(-D / b["sigma"])
    idx  = np.argsort(S, axis=1)[:, -b["k"]:]
    sv   = np.take_along_axis(S, idx, axis=1); lab = b["y"][idx]
    w    = np.exp((sv - sv.max(axis=1, keepdims=True)) / b["tau"]) * b["cls_w"][lab.astype(int)]
    return pd.DataFrame({"M3": (w * lab).sum(axis=1) / w.sum(axis=1)}, index=prof.index)

# ---------- full pipeline prediction for any unit (patient or single visit) ----------
def predict_pipeline(B, X, units):
    """Scores any unit: pass PATNO for patient level, or np.arange(len(X)) for single visits."""
    X   = X[B["features"] + DEMO]
    fp  = unit_mean(units, m1_feature_probs(B["m1"], X))[B["features"]]
    out = pd.DataFrame(index=fp.index)
    out["M1"] = predict_xgb_stacker(B["stacker"], fp.values)
    out["M2"] = unit_mean(units, {"p": predict_m2(B["m2"], X)})["p"]
    out = out.join(predict_m3(B["m3"], X, units))
    out["Final"] = predict_lr_on_logits(B["meta"], out[["M1", "M2", "M3"]].values)
    return out


def fold_task(X, y, g, tr_i, va_i, features, seed):
    # harmonise the training rows of this fold only; validation rows stay untouched
    Xtr = harmonize(X.iloc[tr_i], y[tr_i], g[tr_i], features, seed)
    Xva = X.iloc[va_i]
    m1 = {f: fit_m1_feature(Xtr[[f, "age", "sex"]], y[tr_i]) for f in features}
    return {
        "va_i": va_i,
        "m1":   m1_feature_probs(m1, Xva),
        "m2":   predict_m2(fit_m2(Xtr, y[tr_i]), Xva),
        "m3":   predict_m3(fit_m3(Xtr, y[tr_i], g[tr_i]), Xva, g[va_i])["M3"],
    }

def run_oof(X, y, g, features, n_jobs=-1):
    splits = [(tr_i, va_i) for r in range(N_REPEATS)
              for tr_i, va_i in StratifiedGroupKFold(N_SPLITS, shuffle=True,
                                                     random_state=RANDOM_STATE + r).split(X, y, g)]
    for tr_i, va_i in splits:                                   # guarantee: no patient on both sides
        assert not set(g[tr_i]) & set(g[va_i])
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(fold_task)(X, y, g, a, b, features, RANDOM_STATE + k)
        for k, (a, b) in enumerate(splits))

    n = len(X)
    s1, s2, cnt = {f: np.zeros(n) for f in features}, np.zeros(n), np.zeros(n)
    m3_parts = []
    for r in res:
        for f in features:
            s1[f][r["va_i"]] += r["m1"][f]
        s2[r["va_i"]] += r["m2"]; cnt[r["va_i"]] += 1
        m3_parts.append(r["m3"])
    assert (cnt == N_REPEATS).all()

    feat_p = unit_mean(g, {f: s1[f] / cnt for f in features})[features]   # patient x feature
    oof = pd.DataFrame(index=feat_p.index)
    oof["M2"] = unit_mean(g, {"p": s2 / cnt})["p"]
    oof["M3"] = pd.concat(m3_parts).groupby(level=0).mean()
    oof["y"]  = pd.Series(y).groupby(g).first()
    return feat_p, oof


def train_final(X, y, g, feat_p, oof, features, config, n_jobs=-1):
    yp = oof["y"].values
    X = harmonize(X, y, g, features, RANDOM_STATE)   # final models: harmonised training rows
    fitted = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(fit_m1_feature)(X[[f, "age", "sex"]], y) for f in features)
    B = {"m1": dict(zip(features, fitted)),
         "stacker": fit_xgb_stacker(feat_p.values, yp),
         "m2": fit_m2(X, y),
         "m3": fit_m3(X, y, g)}
    # OOF M1 for the meta model must itself be out-of-fold -> cross-fit the stacker
    oof["M1"] = crossfit(fit_xgb_stacker, predict_xgb_stacker, feat_p.values, yp, seed_offset=100)
    B["meta"] = fit_lr_on_logits(oof[["M1", "M2", "M3"]].values, yp)
    oof["Final"] = crossfit(fit_lr_on_logits, predict_lr_on_logits,
                            oof[["M1", "M2", "M3"]].values, yp, seed_offset=200)
    # Decision threshold: Youden J on TRAINING OOF predictions, then frozen
    fpr, tpr, thr = roc_curve(yp, oof["Final"])
    B["threshold"] = float(thr[np.argmax(tpr - fpr)])
    B["features"], B["config"] = list(features), config
    return B


def _cal(y, p):
    """Calibration intercept (calibration-in-the-large, slope fixed at 1) and calibration slope."""
    lp = _lg(p); a = 0.0
    for _ in range(50):                                          # Newton-Raphson with offset lp
        q = expit(a + lp); h = (q * (1 - q)).sum()
        if h < 1e-12: break
        a += (y - q).sum() / h
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y).coef_[0, 0]
    return a, slope

def point_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    ppv, npv   = tp / max(tp + fp, 1), tn / max(tn + fn, 1)
    ci, cs     = _cal(y, p)
    return {"AUC": roc_auc_score(y, p), "Sensitivity": sens, "Specificity": spec,
            "Balanced_Acc": (sens + spec) / 2, "PPV": ppv, "NPV": npv,
            "F1": 2 * tp / max(2 * tp + fp + fn, 1), "Brier": np.mean((p - y) ** 2),
            "Cal_intercept": ci, "Cal_slope": cs, "TP": tp, "FP": fp, "TN": tn, "FN": fn}

def boot_metrics(y, p, thr, clusters, seed=RANDOM_STATE):
    """Point estimate + 95% percentile CI, resampling whole patients (clusters)."""
    y, p, clusters = np.asarray(y), np.asarray(p), np.asarray(clusters)
    pt = point_metrics(y, p, thr)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng, draws = np.random.RandomState(seed), []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[i] for i in rng.randint(0, len(uniq), len(uniq))])
        if len(np.unique(y[idx])) < 2: continue
        draws.append(point_metrics(y[idx], p[idx], thr))
    dr = pd.DataFrame(draws)
    row = {}
    for k in ["AUC", "Sensitivity", "Specificity", "Balanced_Acc", "PPV", "NPV", "F1", "Brier",
              "Cal_intercept", "Cal_slope"]:
        lo, hi = np.percentile(dr[k], [2.5, 97.5])
        row[k] = pt[k]; row[k + "_CI"] = f"[{lo:.3f}, {hi:.3f}]"
    row.update({k: int(pt[k]) for k in ["TP", "FP", "TN", "FN"]})
    return row

def evaluate(pred, y, clusters, thr, label, n_jobs=-1):
    """One bootstrap job per model column, run in parallel."""
    cols = ["M1", "M2", "M3", "Final"]
    cols = [c for c in cols if c in pred]
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(boot_metrics)(y, pred[c].values, t, clusters)
        for c in cols for t in ([thr, 0.5] if c == "Final" else [0.5]))
    names = [(c, t) for c in cols for t in ([thr, 0.5] if c == "Final" else [0.5])]
    df = pd.DataFrame(rows)
    df.insert(0, "Threshold", [round(t, 3) for _, t in names])
    df.insert(0, "Model", [c for c, _ in names])
    df.insert(0, "Evaluation", label)
    return df

def auc_diff_ci(y, pa, pb, clusters, seed=RANDOM_STATE):
    """Paired cluster-bootstrap CI for AUC(a) - AUC(b)."""
    y, pa, pb, clusters = map(np.asarray, (y, pa, pb, clusters))
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng, d = np.random.RandomState(seed), []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[i] for i in rng.randint(0, len(uniq), len(uniq))])
        if len(np.unique(y[idx])) < 2: continue
        d.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    d = np.array(d)
    return roc_auc_score(y, pa) - roc_auc_score(y, pb), *np.percentile(d, [2.5, 97.5]), 2 * min((d <= 0).mean(), (d >= 0).mean())

