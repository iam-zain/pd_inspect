"""Summarise the harmonisation candidates into comparison.csv / comparison.md."""
import os, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
rows = []
for c in sorted(os.listdir(R)):
    f = os.path.join(R, c)
    if not os.path.exists(os.path.join(f, "info.json")):
        continue
    info = json.load(open(os.path.join(f, "info.json")))
    m = pd.read_csv(os.path.join(f, "metrics.csv")); st = pd.read_csv(os.path.join(f, "stress_tests.csv"))
    cmp_ = pd.read_csv(os.path.join(f, "auc_comparison.csv"))
    fin = lambda ev: m[(m.Evaluation == ev) & (m.Model == "Final") & (m.Threshold != 0.5)].iloc[0]
    sub = lambda ev, mod: m[(m.Evaluation == ev) & (m.Model == mod)].iloc[0]
    s = lambda grp, cond: 100 * st[(st.group == grp) & (st.cond == cond)].calledPD.iloc[0]
    a, o, v, tr = fin("TEST patient, all"), fin("TEST patient, original cohort"), fin("TEST single visit, all"), fin("train OOF (patient)")
    nc = info["negative_controls"]
    d3 = cmp_[(cmp_.Test == "original") & (cmp_.Comparison == "Final - M3")].iloc[0]
    rows.append({
        "Candidate": c, "Harmonisation": info["mode"], "Training cohort": "original" if info["max_patno"] else "all",
        "Values removed %": round(nc["observed_values_removed_pct"], 2),
        "NegCtl share LR": round(nc["share_LR"], 3), "NegCtl share XGB": round(nc["share_XGB"], 3), "NegCtl never-recorded XGB": round(nc["never_XGB"], 3),
        "Never Thermo PD/HC %": f"{100*nc['never_thermo_PD']:.1f}/{100*nc['never_thermo_HC']:.1f}",
        "Threshold": round(info["threshold"], 3),
        "Train OOF AUC": round(tr.AUC, 3),
        "Test AUC (all, n=181)": f"{a.AUC:.3f} {a.AUC_CI}", "Sens (all)": round(a.Sensitivity, 3), "Spec (all)": round(a.Specificity, 3),
        "BalAcc (all)": round(a.Balanced_Acc, 3), "Brier (all)": round(a.Brier, 3), "Cal slope (all)": round(a.Cal_slope, 3),
        "Test AUC (orig, n=123)": f"{o.AUC:.3f} {o.AUC_CI}", "Sens (orig)": round(o.Sensitivity, 3), "Spec (orig)": round(o.Specificity, 3),
        "Brier (orig)": round(o.Brier, 3), "Single-visit AUC": round(v.AUC, 3),
        "M1/M2/M3 AUC (orig)": "/".join(f"{sub('TEST patient, original cohort', k).AUC:.3f}" for k in ["M1", "M2", "M3"]),
        "Final-M3 dAUC (orig)": f"{d3.dAUC:+.3f} [{d3.lo:.3f}, {d3.hi:.3f}] p={d3.p:.3f}",
        "Meta w M1/M2/M3": "/".join(f"{w:.2f}" for w in info["meta_coef"]),
        "All-blank P(final)": round(info["allblank"]["Final"], 3), "All-blank P(M3)": round(info["allblank"]["M3"], 3),
        "HC called PD %: orig / matched / cohort-mask / SCOPA-3": f"{s('HC','orig'):.1f} / {s('HC','matched_random'):.1f} / {s('HC','cohort_mask'):.1f} / {s('HC','scopa3_removed'):.1f}",
        "PD(orig) called PD %: orig / matched / cohort-mask / SCOPA-3": f"{s('PD_orig','orig'):.1f} / {s('PD_orig','matched_random'):.1f} / {s('PD_orig','cohort_mask'):.1f} / {s('PD_orig','scopa3_removed'):.1f}",
        "HC shift cohort-mask minus matched (pp)": round(s("HC", "cohort_mask") - s("HC", "matched_random"), 1),
        "PD shift SCOPA-3 minus orig (pp)": round(s("PD_orig", "scopa3_removed") - s("PD_orig", "orig"), 1),
    })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "comparison.csv"), index=False)
with open(os.path.join(HERE, "comparison.md"), "w", encoding="utf8") as fh:
    fh.write(df.set_index("Candidate").T.to_markdown())
print(df.set_index("Candidate").T.to_string())
