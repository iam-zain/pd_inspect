# M3 variants: results (local run, scikit-learn 1.7.2 folds)

The 12 M3 collaborative-filtering variants (Table 5 of the article) were run inside the final pipeline: same primary
cohort (PATNO < 10000), same harmonisation in every fold, same 5 x 3 StratifiedGroupKFold, same M1 and M2,
same L2 logistic-regression meta-learner, threshold = Youden index on the training out-of-fold predictions.
Selection uses the training out-of-fold results (501 participants); the test set (123 participants: 83 PD, 40 HC) is
used for confirmation only.

- `top12_m3_variants.csv`: the 12 variants ranked by ensemble OOF AUC. `table5_label` gives the label used in the
  article (V1-V12; the final M3 is V4: Euclidean RBF similarity, LR-weighted features, K = 80, softmax temperature 0.05,
  class re-weighted votes).
- `../M3_Redesign_Analysis.ipynb` reproduces this file (`QUICK = True` for a smoke test). The M1 and M2 predictions
  are those of the local run, and the M3 predictions of V4 are identical to `results/primary/oof_predictions_train.csv`.
