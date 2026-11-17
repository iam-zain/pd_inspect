import os

# ── Limit threads BEFORE any numpy/sklearn/xgb import (PythonAnywhere OOM fix) ──
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json
import logging
import random
import time
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_file

# ── Logging (SAFE for PythonAnywhere) ─────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model loading ─────────────────────────────────────────────────
# One bundle now holds M1 + M2 + M3 + meta + the feature list + the decision threshold.
# It is produced by Build_Prevent_Model/no_UPDRS/Final_Model/Train_3SubModels.ipynb.
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

try:
    from joblib import load as joblib_load

    BUNDLE    = joblib_load(os.path.join(MODEL_DIR, "pipeline.joblib"))
    FEATURES  = BUNDLE["features"]
    THRESHOLD = float(BUNDLE["threshold"])

    # Keep every booster single-threaded (prevents OOM on constrained workers)
    for _b in list(BUNDLE["m1"].values()) + [BUNDLE["m2"]]:
        _xgb = _b.get("xgb")
        if _xgb is not None and hasattr(_xgb, "set_params"):
            try:
                _xgb.set_params(n_jobs=1, nthread=1)
            except Exception:
                pass

    MODELS_LOADED = True
    logger.info("Loaded Min16 pipeline: %d features, threshold %.3f", len(FEATURES), THRESHOLD)

except Exception:
    logger.error("Model loading failed", exc_info=True)
    MODELS_LOADED = False
    BUNDLE, FEATURES, THRESHOLD = None, [], 0.5

# ── Real-patient sample bank (for the "Random 16 / Random 30" quick-fill buttons) ──
# Genuine PPMI train/test visit rows (>=16 answered features, Smell + Gastrointestinal
# both recorded), split by true class. No patient identifiers are stored - only the
# clinical scores, age and sex. Built by Final_Model/build_sample_bank.py.
try:
    with open(os.path.join(MODEL_DIR, "sample_bank.json")) as _fh:
        SAMPLE_BANK = json.load(_fh)
    logger.info("Loaded sample bank: %d healthy, %d PD real records",
                len(SAMPLE_BANK.get("healthy", [])), len(SAMPLE_BANK.get("pd", [])))
except FileNotFoundError:
    logger.warning("sample_bank.json not found; the quick-fill buttons are disabled")
    SAMPLE_BANK = {"healthy": [], "pd": []}
except Exception:
    logger.error("Sample bank loading failed", exc_info=True)
    SAMPLE_BANK = {"healthy": [], "pd": []}

from prediction import predict_patient, InsufficientInput, MIN_ANSWERED
from pdf_report import generate_report

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", min_features=MIN_ANSWERED,
                           n_features=len(FEATURES))


@app.route("/box")
def box():
    return render_template("box.html")


# ─────────────────────────────────────────────────────────────────
@app.route("/features")
def features():
    """Feature list the UI offers. Driven by the model, so the two cannot drift apart."""
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 503
    return jsonify({"features": list(FEATURES), "minFeatures": MIN_ANSWERED,
                    "threshold": THRESHOLD})


# ─────────────────────────────────────────────────────────────────
@app.route("/sample_patient")
def sample_patient():
    """One real, de-identified PPMI record (train/test set) for the quick-fill buttons.

    Every value returned is a genuine recorded answer - nothing here is generated or
    perturbed. `cls` selects which real class to draw from; the frontend is
    responsible for the 3:2 Healthy:PD draw ratio.
    """
    cls = request.args.get("cls", "healthy").strip().lower()
    key = "healthy" if cls == "healthy" else "pd"
    pool = SAMPLE_BANK.get(key) or []
    if not pool:
        return jsonify({"error": "Sample bank not available"}), 503
    return jsonify(random.choice(pool))


# ─────────────────────────────────────────────────────────────────
@app.route("/submit", methods=["POST"])
def submit():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 503

    t0 = time.time()

    try:
        age    = float(request.form.get("Age", ""))
        gender = int(request.form.get("Gender", ""))
        n_feat = int(request.form.get("num_features", "0"))
    except Exception:
        return jsonify({"error": "Invalid Age / Sex / num_features"}), 400

    input_dict = {"age": age, "sex": gender}

    for i in range(1, n_feat + 1):
        feat_name  = request.form.get(f"Feature{i}", "").strip()
        feat_value = request.form.get(f"Score_Feature{i}", "")

        if not feat_name or feat_name not in FEATURES:
            continue

        try:
            input_dict[feat_name] = float(feat_value)
        except (TypeError, ValueError):
            pass

    try:
        final_prob, risk_label, confidence, breakdown = predict_patient(input_dict, BUNDLE)
    except InsufficientInput as exc:
        # Below the inclusion rule the model was validated under: refuse, do not guess.
        return jsonify({"error": str(exc), "insufficientInput": True,
                        "minFeatures": MIN_ANSWERED}), 400
    except Exception as exc:
        import traceback
        err_detail = traceback.format_exc()
        logger.error("Prediction failed: %s", err_detail)
        return jsonify({"error": "Prediction failed", "detail": str(exc),
                        "trace": err_detail}), 500

    logger.info("/submit completed in %.3fs (n_features=%s)", time.time() - t0,
                len(breakdown["used_features"]))

    user_status      = "PROFILE_1" if "High" in risk_label else "PROFILE_2"
    confidence_score = round(confidence["combined_score"] * 100)

    return jsonify({
        "userStatus":        user_status,
        "riskLabel":         risk_label,
        "confidenceScore":   confidence_score,
        "confidenceBand":    confidence["band"],
        "probabilityMargin": confidence["probability_margin"],
        "modelConsensus":    confidence["model_consensus"],
        "subModelVotes":     confidence["sub_model_votes"],
        "finalProb":         breakdown["final_prob"],
        "threshold":         breakdown["threshold"],
        "probM1":            breakdown["prob_m1"],
        "probM2":            breakdown["prob_m2"],
        "probM3":            breakdown["prob_m3"],
        "usedFeatures":      breakdown["used_features"],
        "defaultedFeatures": breakdown["defaulted_features"],
        "numProvided":       len(breakdown["used_features"]),
    })


# ─────────────────────────────────────────────────────────────────
@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    data = request.get_json(force=True, silent=True)

    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    try:
        pdf_bytes = generate_report(data)
    except Exception:
        logger.error("PDF generation failed", exc_info=True)
        return jsonify({"error": "PDF generation failed"}), 500

    raw_name      = (data.get("patientName") or "").strip().replace(" ", "_")
    download_name = f"PD_INSPECT_Report_{raw_name}.pdf" if raw_name else "PD_INSPECT_Report.pdf"

    return send_file(
        BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=download_name,
    )


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
