"""
Prediction for the Min16 robust pipeline (30 NMS features, no MDS-UPDRS Part I, no Hand).

The model bundle (saved_models/pipeline.joblib) is produced by
Build_Prevent_Model/no_UPDRS/Final_Model/Train_3SubModels.ipynb and contains
M1 (symptom-wise ensemble: per-item LR + SVM + XGB, XGBoost stacker), M2 (joint-symptom XGBoost), M3 (participant-profile collaborative filtering),
the meta model, the feature list and the decision threshold chosen on training data.

Key differences from the previous version of this file:
  * Blanks are filled with class-balanced values stored inside the bundle, so a missing
    item no longer pushes the prediction toward PD.
  * The decision threshold comes from the bundle (training-OOF Youden J), not a fixed 0.5.
  * At least MIN_ANSWERED of the 30 features are required; the app refuses to score
    sparser input instead of returning a confident-looking number.
"""
import numpy as np
import pandas as pd

from min16_lib import predict_pipeline

MIN_ANSWERED = 16          # inclusion rule the model was trained and validated under


class InsufficientInput(ValueError):
    """Raised when fewer than MIN_ANSWERED clinical features were provided."""


def predict_patient(input_dict, bundle):
    """
    Score one patient.

    input_dict : {"age": float, "sex": 0|1, "<Feature>": value, ...}
                 Features may be missing; they are imputed with class-balanced values.

    Returns (final_prob, risk_label, confidence, breakdown).
    """
    features  = bundle["features"]
    threshold = float(bundle["threshold"])

    provided = [f for f in features
                if f in input_dict and input_dict[f] is not None and not pd.isna(input_dict[f])]
    if len(provided) < MIN_ANSWERED:
        raise InsufficientInput(
            f"{len(provided)} of {len(features)} features provided; "
            f"at least {MIN_ANSWERED} are required for a reliable prediction.")

    row = {f: pd.to_numeric(input_dict.get(f, np.nan), errors="coerce") for f in features}
    X = pd.DataFrame([row], columns=features).astype(float)
    X["age"] = float(input_dict.get("age", np.nan))
    X["sex"] = float(input_dict.get("sex", np.nan))

    pred = predict_pipeline(bundle, X, np.array([0])).iloc[0]
    prob1, prob2, prob3 = float(pred["M1"]), float(pred["M2"]), float(pred["M3"])
    final_prob  = float(pred["Final"])
    final_class = int(final_prob >= threshold)

    risk_label = "High Risk (PD)" if final_class else "Low Risk (Healthy)"

    # ── Confidence: distance from the threshold + agreement between sub-models ──
    # Scaled within the side of the threshold the prediction falls on, so that a
    # probability near 0 or 1 reads as confident even though the threshold is not 0.5.
    span      = (1.0 - threshold) if final_class else threshold
    margin    = min(abs(final_prob - threshold) / span, 1.0) if span > 0 else 1.0
    sub_votes = [int(p >= threshold) for p in (prob1, prob2, prob3)]
    consensus = sum(1 for v in sub_votes if v == final_class) / 3.0
    combined  = 0.70 * margin + 0.30 * consensus

    if combined >= 0.80:
        conf_band = "High Confidence"
    elif combined >= 0.55:
        conf_band = "Moderate Confidence"
    elif combined >= 0.35:
        conf_band = "Low Confidence"
    else:
        conf_band = "Very Low Confidence — treat with caution"

    confidence = {
        "combined_score": round(combined, 4),
        "band": conf_band,
        "probability_margin": round(margin, 4),
        "model_consensus": round(consensus, 4),
        "sub_model_votes": {
            "M1 (Symptom-wise)": "PD" if prob1 >= threshold else "Healthy",
            "M2 (Joint-symptom)":    "PD" if prob2 >= threshold else "Healthy",
            "M3 (Participant-profile)": "PD" if prob3 >= threshold else "Healthy",
        },
    }

    breakdown = {
        "prob_m1": round(prob1, 4),
        "prob_m2": round(prob2, 4),
        "prob_m3": round(prob3, 4),
        "final_prob": round(final_prob, 4),
        "threshold": round(threshold, 4),
        "used_features": provided,
        "defaulted_features": [f for f in features if f not in provided],
    }

    return final_prob, risk_label, confidence, breakdown
