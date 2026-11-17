// Minimum number of features the model was validated on. Overwritten by /features.
var MIN_FEATURES = 16;
fetch('/features').then(function(r){return r.json();}).then(function(d){
  if (d && d.minFeatures) {
    MIN_FEATURES = d.minFeatures;
    document.querySelectorAll('.min-features').forEach(function(el){ el.innerText = MIN_FEATURES - 1; });
    updateSubmitReadiness();   // the threshold may have moved
  }
}).catch(function(){});

const config = window.factorConfig;

// ── Back-to-top button: shows once the page is scrolled down a bit ───────────
window.addEventListener("scroll", function() {
  const btn = document.getElementById("backToTopBtn");
  if (btn) btn.classList.toggle("pd-visible", window.scrollY > 300);
});

// ── Globals to hold latest result for PDF/report ─────────────────────────────
let _lastResult   = null;   // server JSON
let _lastFormData = null;   // FormData used for /submit
let _factorCount  = 0;

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────

const getRiskCategoryMessage = (riskCategory) =>
  riskCategory === "PROFILE_1"
    ? `<span style="color:#F07569;font-size:1.4em;font-weight:700;">&#9888; High Risk (PD)</span>`
    : `<span style="color:#4FA77F;font-size:1.4em;font-weight:700;">&#10003; Low Risk (Healthy)</span>`;

const getConfidenceBandColor = (band) => {
  if (band === "High Confidence")     return "#84CDAC";
  if (band === "Moderate Confidence") return "#E3A687";
  return "#F07569";
};

const miniBar = (prob, color, width) => {
  width = width || 160;
  const pct = Math.round(prob * 100);
  return '<div style="display:inline-flex;align-items:center;gap:6px;">'
    + '<span style="min-width:34px;font-size:12px;">' + pct + '%</span>'
    + '<div style="background:#e8e8e8;border-radius:4px;overflow:hidden;width:' + width + 'px;height:10px;">'
    + '<div style="background:' + color + ';width:' + pct + '%;height:100%;border-radius:4px;"></div>'
    + '</div></div>';
};

// Every factor label in factorsConfig.js reads "Score of <feature name>". Splitting
// that off lets the feature name itself stand out instead of "Score of" repeating
// on every row.
const SCORE_PREFIX_RE = /^(Score of)\s+(.*)$/i;

// Drop the "Score of" prefix entirely - used where space is tight (sidebar summary).
const stripScorePrefix = function(label) {
  const m = SCORE_PREFIX_RE.exec(label);
  return m ? m[2] : label;
};

// Keep "Score of" but de-emphasise it, so the feature name reads first - used where
// the full label is still wanted (the slider row).
const dimScorePrefixHTML = function(label) {
  const m = SCORE_PREFIX_RE.exec(label);
  if (!m) return label;
  return '<span style="opacity:.55;font-weight:400;">' + m[1] + '</span> '
    + '<span style="font-weight:700;">' + m[2] + '</span>';
};

// ─────────────────────────────────────────────────────────────────────────────
// SUMMARY SIDEBAR
// ─────────────────────────────────────────────────────────────────────────────

const getSummaryItemHTML = function(item) {
  return '<li id="summary' + item.id + '" class="list-group-item d-flex justify-content-between lh-condensed">'
    + '<div><h6 class="my-0">' + stripScorePrefix(item.label) + topFeatureBadgeHTML(item.id)
    + ' <span data-toggle="tooltip" title="About" data-placement="top" class="info-feature pd-icon-btn pd-icon-sm" role="button" aria-label="About this factor" onclick="showInfo(' + item.id + ')">'
    + '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 7.2v4.3"/><circle cx="8" cy="4.6" r="0.4"/></svg></span>'
    + '</h6>'
    + (item.description ? '<small class="text-muted">' + item.description + '</small>' : '')
    + '</div><span class="text-muted" id="summaryVal' + item.id + '">' + item.value + '</span></li>';
};

// ─────────────────────────────────────────────────────────────────────────────
// TOP-FEATURE BADGE (Final-ensemble SHAP top 10 - see static/topShapFeatures.js)
// ─────────────────────────────────────────────────────────────────────────────

const isTopShapFeature = function(key) {
  return (window.topShapFeatures || []).indexOf(key) !== -1;
};

const TOP_FEATURE_TOOLTIP = "One of the strongest predictors in our study.";

// Small inline-styled badge (this app has no shared stylesheet for its JS-built
// markup, so every other snippet in this file is inline-styled too - matched here).
// Full "Top Predictor" pill - used once a factor is added/selected, where there is
// room for it: the slider row label, the sidebar summary, the selected-item box.
const topFeatureBadgeHTML = function(key) {
  if (!isTopShapFeature(key)) return "";
  return ' <span data-toggle="tooltip" title="' + TOP_FEATURE_TOOLTIP + '"'
    + ' style="display:inline-block;background:#FBF0DC;color:#8A6A2E;border:1px solid #F2CC8F;'
    + 'border-radius:10px;padding:1px 7px;font-size:10.5px;font-weight:700;vertical-align:middle;'
    + 'white-space:nowrap;">&#11088; Top Predictor</span>';
};

// Compact star only - used in the dropdown list, where the full pill wraps onto a
// second line and looks broken against long labels.
const topFeatureStarHTML = function(key) {
  if (!isTopShapFeature(key)) return "";
  return ' <span data-toggle="tooltip" title="' + TOP_FEATURE_TOOLTIP + '"'
    + ' style="color:#E3A687;font-size:13px;vertical-align:middle;">&#11088;</span>';
};

// Restore the "Select a test or symptom" placeholder without wiping the caret icon
// that lives alongside the label inside .selected-item.
const resetSelectedItemBox = function() {
  const box = $(".selected-item")[0];
  box.querySelector(".selected-item-label").innerText = "Select a test or symptom ";
  delete box.dataset.key;
};

// Grey out (and block clicks on) a factor's dropdown entry once it has been added,
// so it is obvious at a glance which tests are already selected. Re-enabled when
// the factor is removed.
const setDropdownItemDisabled = function(key, disabled) {
  const el = document.querySelector('.select-item[data-key="' + key + '"]');
  if (!el) return;
  el.style.opacity = disabled ? "0.45" : "";
  el.style.pointerEvents = disabled ? "none" : "";
  el.style.cursor = disabled ? "default" : "";
  el.setAttribute("aria-disabled", disabled ? "true" : "false");
};

// ─────────────────────────────────────────────────────────────────────────────
// FEATURE SLIDER ROW
// ─────────────────────────────────────────────────────────────────────────────

const getItemHTML = function(item) {
  const step = item.step || 1;
  return '<div class="form-group mb-4" id="container' + item.id + '">'
    + '<label for="' + item.id + '">' + dimScorePrefixHTML(item.label) + topFeatureBadgeHTML(item.id)
    + ' <span class="text-muted small">(min: ' + item.min + ', max: ' + item.max + ')</span>'
    + ' <span data-toggle="tooltip" title="Remove this factor" data-placement="top" class="remove-feature pd-icon-btn" role="button" aria-label="Remove this factor" onclick="removeFeature(' + item.id + ')">'
    + '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/></svg></span>'
    + ' <span data-toggle="tooltip" title="About" data-placement="top" class="info-feature pd-icon-btn" role="button" aria-label="About this factor" onclick="showInfo(' + item.id + ')">'
    + '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 7.2v4.3"/><circle cx="8" cy="4.6" r="0.4"/></svg></span>'
    + '</label>'
    + '<div class="row">'
    + '<div class="col-9 d-flex">'
    + '<input id="' + item.id + '" min="' + item.min + '" max="' + item.max + '" type="range"'
    + ' step="' + step + '" value="' + item.value + '"'
    + ' class="form-control-range factors-input" oninput="handleRangeChange(this);" />'
    + '</div>'
    + '<div class="col-3 d-flex">'
    + '<input min="' + item.min + '" max="' + item.max + '" type="number"'
    + ' step="' + step + '" value="' + item.value + '" required'
    + ' id="' + item.id + '_current"'
    + ' class="form-control factors-input factors-input-box"'
    + ' onchange="handleRangeInput(this, ' + item.id + ');" />'
    + '</div></div></div>';
};

// ─────────────────────────────────────────────────────────────────────────────
// INFO MODAL
// ─────────────────────────────────────────────────────────────────────────────

const getInfoModalBody = function(item) {
  return '<p>' + item.info + '</p><img src="' + item.img + '" class="info-img"/>';
};

// ─────────────────────────────────────────────────────────────────────────────
// RESULT MODAL — SIMPLE VIEW (user-friendly)
// ─────────────────────────────────────────────────────────────────────────────

const getSimpleResultHTML = function(r, userName, nFeatures) {
  const bandColor   = getConfidenceBandColor(r.confidenceBand);
  const isPD        = r.userStatus === "PROFILE_1";
  const display_pct = isPD ? Math.round(r.finalProb * 100)
                            : Math.round((1 - r.finalProb) * 100);
  const pd_pct      = Math.round(r.finalProb * 100);
  const hl_pct      = 100 - pd_pct;
  const conf_pct    = r.confidenceScore;
  const name        = (userName && userName.trim()) ? userName.trim() : "the patient";
  const feats       = nFeatures || r.numProvided || "several";

  const sentence = isPD
    ? "Based on the analysis of " + feats + " features, the model predicts that "
      + name + " has a <b style=\"color:#F07569;\">" + pd_pct + "% probability</b> "
      + "of having Parkinson&#39;s Disease (High Risk)."
    : "Based on the analysis of " + feats + " features, the model predicts that "
      + name + " has an <b style=\"color:#4FA77F;\">" + hl_pct + "% probability</b> "
      + "of being Healthy, indicating Low Risk of Parkinson&#39;s Disease.";

  return ''
    + '<div style="text-align:center;padding:20px 10px 6px;">'
    + getRiskCategoryMessage(r.userStatus)
    + '<div style="margin:14px auto;width:180px;">'
    + '<div id="resultProgressBar" role="progressbar" style="--value:' + display_pct + ';--primary:' + (isPD ? '#F07569' : '#84CDAC') + '"></div>'
    + '</div>'
    + '<div style="font-size:13px;color:#666;margin-top:4px;">'
    + '<b style="color:#F07569;">' + pd_pct + '% PD</b>'
    + '&nbsp;&nbsp;&middot;&nbsp;&nbsp;'
    + '<b style="color:#4FA77F;">' + hl_pct + '% Healthy</b>'
    + '</div>'
    + '<div style="width:90%;margin:12px auto 0;font-size:13px;color:#555;line-height:1.6;background:#f8f9fa;border-radius:8px;padding:12px 16px;">'
    + sentence
    + '</div>'
    + '</div>'

    + '<hr style="width:90%;margin:12px auto;"/>'

    + '<div style="width:90%;margin:0 auto 18px;">'
    + '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
    + '<span style="font-size:14px;font-weight:600;color:#333;">Assessment Confidence</span>'
    + '<span style="font-size:14px;font-weight:700;color:' + bandColor + ';">' + r.confidenceBand + '</span>'
    + '</div>'
    + '<div style="background:#e8e8e8;border-radius:8px;overflow:hidden;height:14px;">'
    + '<div style="background:' + bandColor + ';width:' + conf_pct + '%;height:100%;border-radius:8px;"></div>'
    + '</div>'
    + '<div style="display:flex;justify-content:space-between;font-size:11px;color:#aaa;margin-top:3px;">'
    + '<span>Low</span><span>High</span></div>'
    + '<div style="margin-top:6px;font-size:12px;color:#888;text-align:center;">'
    + 'See &quot;View Full Report&quot; for detailed model breakdown.'
    + '</div></div>'

    + '<div style="width:90%;margin:0 auto 10px;font-size:11.5px;color:#aaa;text-align:center;line-height:1.5;">'
    + 'This tool is for research &amp; screening purposes only and does <em>not</em> constitute a '
    + 'clinical diagnosis. Please consult a qualified neurologist for medical evaluation.'
    + '</div>';
};


// ─────────────────────────────────────────────────────────────────────────────
// RESULT MODAL — FULL REPORT VIEW
// ─────────────────────────────────────────────────────────────────────────────

const getFullReportHTML = function(r) {
  const bandColor = getConfidenceBandColor(r.confidenceBand);
  const isPD      = r.userStatus === "PROFILE_1";
  const pd_pct    = Math.round(r.finalProb * 100);
  const hl_pct    = 100 - pd_pct;

  let voteRows = "";
  Object.entries(r.subModelVotes || {}).forEach(function(entry) {
    const model = entry[0], vote = entry[1];
    const match = (vote === "PD") === isPD;
    const color  = vote === "PD" ? "#F07569" : "#4FA77F";
    const icon   = match ? "&#10003;" : "&#10007;";
    voteRows += '<tr>'
      + '<td style="padding:5px 10px;font-size:13px;">' + icon + ' ' + model + '</td>'
      + '<td style="padding:5px 10px;font-weight:700;color:' + color + ';font-size:13px;">' + vote + '</td>'
      + '</tr>';
  });

  const probRow = function(label, prob, color) {
    return '<tr>'
      + '<td style="padding:4px 10px;font-size:13px;">' + label + '</td>'
      + '<td style="padding:4px 10px;">' + miniBar(prob, color) + '</td>'
      + '</tr>';
  };

  // Feature count note — use numProvided (actual user input count)
  const nProvided   = r.numProvided || 0;
  const nTotal      = (r.usedFeatures||[]).length + (r.defaultedFeatures||[]).length;
  const nDefaulted  = Math.max(0, nTotal - nProvided);
  const usedNote    = nProvided > 0
    ? nProvided + ' feature(s) measured; ' + nDefaulted + ' filled with class-balanced defaults (midpoint of PD and healthy values).'
    : '';

  const finalColor = isPD ? "#F07569" : "#4FA77F";

  return '<div style="padding:10px 5px;width:100%;">'
    + '<div style="text-align:center;border-bottom:2px solid #2E3A59;padding-bottom:12px;margin-bottom:16px;">'
    + '<h5 style="color:#2E3A59;font-weight:700;margin:0;">PD-INSPECT — Full Assessment Report</h5>'
    + '<small style="color:#888;">Three-Model Ensemble &middot; Parkinson&#39;s Disease Risk Evaluation</small>'
    + '</div>'

    + '<div style="text-align:center;margin-bottom:16px;">'
    + getRiskCategoryMessage(r.userStatus)

    + '</div>'

    // Confidence
    + '<div style="background:#f8f9fa;border-radius:8px;padding:12px 16px;margin-bottom:14px;">'
    + '<div style="font-size:13px;font-weight:700;color:' + bandColor + ';margin-bottom:6px;">'
    + 'Confidence: ' + r.confidenceBand + ' (' + r.confidenceScore + '%)</div>'
    + '<table style="width:100%;font-size:12px;">'
    + '<tr><td style="padding:2px 6px;color:#666;">Probability Margin</td>'
    + '<td style="padding:2px 6px;">' + miniBar(r.probabilityMargin, bandColor, 100) + '</td>'
    + '<td style="padding:2px 6px;color:#aaa;font-size:11px;">Distance from the decision threshold</td></tr>'
    + '<tr><td style="padding:2px 6px;color:#666;">Model Consensus</td>'
    + '<td style="padding:2px 6px;">' + miniBar(r.modelConsensus, bandColor, 100) + '</td>'
    + '<td style="padding:2px 6px;color:#aaa;font-size:11px;">Fraction of sub-models agreeing</td></tr>'
    + '</table></div>'

    // Sub-model votes
    + '<div style="margin-bottom:14px;">'
    + '<div style="font-size:13px;font-weight:700;color:#333;margin-bottom:4px;">Sub-Model Votes</div>'
    + '<table style="width:100%;">' + voteRows + '</table></div>'

    // Probabilities — M3 sub-metrics hidden
    + '<div style="margin-bottom:14px;">'
    + '<div style="font-size:13px;font-weight:700;color:#333;margin-bottom:4px;">Per-Model Probabilities P(PD)</div>'
    + '<table style="width:100%;">'
    + probRow("M1  Symptom-wise",  r.probM1,  "#36C2D3")
    + probRow("M2  Joint-symptom",          r.probM2,  "#B05F9B")
    + probRow("M3  Participant-profile",    r.probM3,  "#84CDAC")
    + probRow("&#9733; Final Ensemble", r.finalProb, finalColor)
    + '</table>'
    + (usedNote ? '<div style="margin-top:6px;font-size:11px;color:#aaa;">' + usedNote + '</div>' : '')
    + '</div>'
    + '<hr/>'
    + '<div style="font-size:11.5px;color:#999;margin-bottom:10px;">'
    + '<b>Confidence</b> = 0.70 &times; Probability Margin + 0.30 &times; Model Consensus. '
    + 'M1 = symptom-wise ensemble (LR+SVM+XGB per item) &middot; M2 = joint-symptom XGBoost &middot; M3 = participant-profile collaborative filtering. '+ 'Decision threshold ' + (r.threshold != null ? r.threshold : 0.5) + ', chosen on training data (Youden J), not on the test set.'
    + '</div>'
    + '<div style="font-size:11px;color:#bbb;text-align:center;">'
    + 'PD-INSPECT is an academic research tool and does <em>not</em> constitute a clinical diagnosis. '
    + 'Please consult a qualified neurologist for medical evaluation.'
    + '</div></div>';
};


// ─────────────────────────────────────────────────────────────────────────────
// MODAL FOOTER BUTTONS
// ─────────────────────────────────────────────────────────────────────────────

const buildModalFooter = function(showingFull) {
  return '<button type="button" id="toggleReportBtn" class="btn btn-outline-secondary btn-sm" onclick="toggleFullReport()">'
    + (showingFull ? '&#8592; Back to Summary' : 'View Full Report')
    + '</button> '
    + '<button type="button" class="btn btn-outline-primary btn-sm" onclick="downloadPDF()">'
    + '&#11015; Download PDF</button> '
    + '<button type="button" class="btn btn-secondary btn-sm" data-dismiss="modal">Close</button>';
};

let _showingFullReport = false;

const toggleFullReport = function() {
  _showingFullReport = !_showingFullReport;
  document.getElementById("resultModalBody").innerHTML   = _showingFullReport ? getFullReportHTML(_lastResult) : getSimpleResultHTML(_lastResult, (document.getElementById("username")||{}).value||"", _lastResult.numProvided);
  document.getElementById("resultModalFooter").innerHTML = buildModalFooter(_showingFullReport);
};

// ─────────────────────────────────────────────────────────────────────────────
// PDF DOWNLOAD
// ─────────────────────────────────────────────────────────────────────────────

const downloadPDF = async function() {
  if (!_lastResult || !_lastFormData) return;

  const featuresForPdf = [];
  const n = parseInt(_lastFormData.get("num_features") || "0");
  for (let i = 1; i <= n; i++) {
    const name  = _lastFormData.get("Feature" + i)       || "";
    const score = _lastFormData.get("Score_Feature" + i) || "";
    const label = (config[name] && config[name].label) ? config[name].label : name;
    featuresForPdf.push({ name: name, label: label, score: score });
  }

  const payload = {
    patientName: (document.getElementById("username") || {}).value || "User",
    age:         _lastFormData.get("Age")    || "",
    gender:      _lastFormData.get("Gender") === "0" ? "Male" : "Female",
    features:    featuresForPdf,
    result:      _lastResult,
  };

  const btns = document.querySelectorAll("[onclick='downloadPDF()']");
  btns.forEach(function(b){ b.innerText = "Generating\u2026"; b.disabled = true; });

  try {
    const resp = await fetch("/generate_pdf", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error("PDF generation failed: " + resp.statusText);
    const blob = await resp.blob();
    const url  = URL.createObjectURL(blob);
    const rawName  = ((document.getElementById("username") || {}).value || "").trim();
    const safeName  = rawName.replace(/\s+/g, "_");
    const pdfName   = safeName ? "PD_INSPECT_Report_" + safeName + ".pdf" : "PD_INSPECT_Report.pdf";
    const a    = document.createElement("a");
    a.href = url; a.download = pdfName;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    alert("Could not generate PDF: " + err.message);
  } finally {
    btns.forEach(function(b){ b.innerText = "\u2B07 Download PDF"; b.disabled = false; });
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// SWS ERROR HELPERS
// ─────────────────────────────────────────────────────────────────────────────

const showSWSError = function(opts) {
  document.querySelector("label.text-danger.small.font-weight-bold.error-helper").innerText = opts.message;
  document.querySelector(".selectpicker-wrp").classList.add("error-show");
};
const removeSWSError = function() {
  document.querySelector(".selectpicker-wrp").classList.remove("error-show");
};

// ─────────────────────────────────────────────────────────────────────────────
// ADD FACTOR
// ─────────────────────────────────────────────────────────────────────────────

const addNewFactor = function() {
  const box = $(".selected-item")[0];
  const key = box.dataset.key;
  if (!key || !config[key]) {
    showSWSError({ message: "Please select a value from the dropdown" });
    return;
  }
  removeSWSError();

  if (document.querySelectorAll("#container" + key).length) {
    showSWSError({ message: "This value already exists" });
    return;
  }

  addFactorByKey(key);
};

// Insert one factor row + summary row. `value` is optional; the config default is used
// when it is omitted. Shared by the dropdown and by the random-fill buttons.
// Label used for ordering. Both lists are kept sorted by it, so factors appear
// alphabetically however they were added (dropdown or random fill).
const factorSortLabel = function(key) {
  return ((config[key] && config[key].label) || key).toLowerCase();
};

// First existing element (matching idPrefix) that should come AFTER `key`
// alphabetically, or null when the new factor belongs at the end.
const nextFactorElement = function(idPrefix, key) {
  const label = factorSortLabel(key);
  const rest  = document.querySelectorAll('[id^="' + idPrefix + '"]');
  for (let i = 0; i < rest.length; i++) {
    const otherKey = rest[i].id.slice(idPrefix.length);
    if (!config[otherKey]) continue;
    if (factorSortLabel(otherKey).localeCompare(label) > 0) return rest[i];
  }
  return null;
};

const addFactorByKey = function(key, value) {
  if (!config[key] || document.querySelectorAll("#container" + key).length) return false;
  const item = Object.assign({ id: key }, config[key]);
  item.value = (value !== undefined && value !== null) ? value : defaultScoreFor(key, config[key]);

  // Slider row: before the first later-sorting row, else after the last one
  const nextRow = nextFactorElement("container", key);
  const $row    = $(getItemHTML(item));
  if (nextRow) $row.insertBefore(nextRow);
  else {
    const rows = document.querySelectorAll('[id^="container"]');
    if (rows.length) $row.insertAfter(rows[rows.length - 1]);
    else $row.insertAfter(document.getElementById("swsTopEl"));
  }

  // Sidebar summary row: same ordering; the list container marks the end
  const nextSummary = nextFactorElement("summary", key);
  $(getSummaryItemHTML(item)).insertBefore(
    nextSummary || document.getElementById("listOfRisks"));

  _factorCount++;
  updateCountOnFactorsChange();
  setDropdownItemDisabled(key, true);
  $('[data-toggle="tooltip"]').tooltip({ container: "body" });
  return true;
};

// ─────────────────────────────────────────────────────────────────────────────
// RANDOM FILL (testing helper)
// ─────────────────────────────────────────────────────────────────────────────

// Snap a raw number onto the factor's own step grid, inside its min/max.
const snapToStep = function(cfg, raw) {
  const step = Number(cfg.step) || 1;
  const min  = Number(cfg.min), max = Number(cfg.max);
  const dec  = (String(step).split(".")[1] || "").length;
  const v    = min + step * Math.round((raw - min) / step);
  return Number(Math.min(max, Math.max(min, v)).toFixed(dec));
};

// Healthy-control reference band for a factor (10th-90th percentile of the healthy
// participants in the PPMI training set), clipped to the slider's own limits.
// Returns null when the factor has no reference data.
const healthyBandFor = function(key, cfg) {
  const h = (window.healthyRanges || {})[key];
  if (!h) return null;
  const min = Number(cfg.min), max = Number(cfg.max);
  return {
    lo:  Math.max(min, Math.min(max, h.lo)),
    hi:  Math.max(min, Math.min(max, h.hi)),
    med: Math.max(min, Math.min(max, h.med)),
  };
};

// The default score shown when a factor is added from the dropdown: the healthy
// median, so an untouched form starts from a healthy-looking profile.
const defaultScoreFor = function(key, cfg) {
  const band = healthyBandFor(key, cfg);
  return band ? snapToStep(cfg, band.med) : cfg.value;
};

// Draw one real, de-identified PPMI record from the server-side sample bank.
// `cls` is "healthy" or "pd". Every value in the response is a genuine recorded
// answer - nothing is generated, interpolated or perturbed.
const fetchSamplePatient = async function(cls) {
  const resp = await fetch("/sample_patient?cls=" + encodeURIComponent(cls));
  const data = await resp.json();
  if (!resp.ok || data.error) throw new Error(data.error || "Could not load a sample patient");
  return data;
};

// Replace the current selection with ONE real patient's genuine recorded scores -
// every test that person actually has answered (real PPMI visits never have all 30
// answered, so this is naturally somewhere from 16 up to the high 20s, whatever that
// person has - never padded or capped).
// The draw is 3:2 Healthy:PD (60% healthy / 40% PD) on every click, so repeated
// clicks sample across both classes in that ratio. Smell and Gastrointestinal are
// always among the results, since the sample bank only contains records with both.
const fillRandomFactors = async function() {
  removeSWSError();
  const cls = Math.random() < 0.6 ? "healthy" : "pd";   // 3:2 healthy:PD

  let record;
  try {
    record = await fetchSamplePatient(cls);
  } catch (err) {
    alert("Could not load a real patient record: " + (err.message || err));
    return;
  }

  document.querySelectorAll(".remove-feature").forEach(function(el) { el.click(); });

  const recorded = record.features || {};
  Object.keys(config).forEach(function(key) {
    if (recorded[key] !== undefined && recorded[key] !== null) {
      addFactorByKey(key, recorded[key]);
    }
  });
  // Cascade the new rows in from top to bottom instead of all at once.
  ['[id^="container"].form-group', 'li[id^="summary"]'].forEach(function(sel) {
    document.querySelectorAll(sel).forEach(function(el, i) {
      el.style.animationDelay = Math.min(i * 22, 500) + "ms";
    });
  });

  // Demographics come from the SAME real record - genuine, not re-rolled separately.
  const ageEl = document.getElementById("Age");
  if (ageEl && record.age != null) ageEl.value = record.age;

  const genders = document.querySelectorAll('[name="gender"]');
  if (genders.length && record.sex != null) {
    const wantFemale = Number(record.sex) >= 0.5;   // sex: 0 = male, 1 = female
    genders.forEach(function(g) { g.checked = (g.value.toLowerCase() === "female") === wantFemale; });
  }

  const nameEl = document.getElementById("username");
  if (nameEl && !nameEl.value.trim()) nameEl.value = "User";

  resetSelectedItemBox();
};

// ─────────────────────────────────────────────────────────────────────────────
// RANGE SYNC
// ─────────────────────────────────────────────────────────────────────────────

const handleRangeChange = function(props) {
  document.querySelector("#" + props.id + "_current").value       = props.value;
  document.getElementById("summaryVal" + props.id).innerText = props.value;
};

const handleRangeInput = function(event, el) {
  if (event.stopPropagation) { event.stopPropagation(); event.preventDefault(); }
  else if (window.event)     { window.event.cancelBubble = true; }
  const min = config[el.id].min, max = config[el.id].max;
  if (Number(event.value) > Number(max)) { alert("Input out of range"); setTimeout(function(){ event.value = max; }, 10); }
  if (Number(event.value) < Number(min)) { alert("Input out of range"); setTimeout(function(){ event.value = min; }, 10); }
  document.getElementById(el.id).value = event.value;
  document.getElementById("summaryVal" + el.id).innerText = event.value;
};

// ─────────────────────────────────────────────────────────────────────────────
// DOCUMENT READY
// ─────────────────────────────────────────────────────────────────────────────

$(document).ready(function() {
  $('[data-toggle="tooltip"]').tooltip({ container: "body" });

  if (window.CSS && window.CSS.registerProperty) {
    window.CSS.registerProperty({ name: "--percentage", syntax: "<integer>", inherits: true, initialValue: "0" });
  }

  const frag = new DocumentFragment();
  Object.keys(config).forEach(function(key) {
    const item = config[key];
    const el = document.createElement("div");
    el.classList.add("select-item");
    el.dataset.key = key;
    el.innerHTML = '<span class="select-item-label">' + item.label + '</span>' + topFeatureStarHTML(key);
    frag.append(el);
  });
  $("#selectOptionsContainer").append(frag);
  $('[data-toggle="tooltip"]').tooltip({ container: "body" });

  setTimeout(function() {
    registerSelectListeners();
    registerResetFormListener();
    setContainerHeightObserver();
    updateSubmitReadiness();   // start dulled: no factors added yet
  }, 10);
});

// ─────────────────────────────────────────────────────────────────────────────
// LISTENERS
// ─────────────────────────────────────────────────────────────────────────────

const registerResetFormListener = function() {
  // The old Reset button was removed in favour of "Clear all" next to the quick-fill
  // buttons. Kept guarded so a page that still has one keeps working.
  const el = document.getElementById("resetButton");
  if (el) el.addEventListener("click", resetForm);
};

// Hide the options list (selecting an item, clicking outside the box, or pressing
// Escape all funnel through here).
const closeFactorDropdown = function() {
  document.querySelector(".select-items").classList.add("hidden");
};

const registerSelectListeners = function() {
  document.querySelector(".selected-item").addEventListener("click", function() {
    this.nextElementSibling.classList.toggle("hidden");
  });
  document.querySelectorAll(".select-item").forEach(function(item) {
    item.addEventListener("click", function() {
      const box = $(".selected-item")[0];
      box.querySelector(".selected-item-label").innerHTML =
        this.querySelector(".select-item-label").textContent + topFeatureBadgeHTML(this.dataset.key);
      box.dataset.key = this.dataset.key;
      closeFactorDropdown();
    });
  });
  document.addEventListener("click", function(event) {
    const box = document.querySelector(".select-box");
    if (!box.contains(event.target)) closeFactorDropdown();
  });
  document.addEventListener("keydown", function(event) {
    if (event.key !== "Escape" && event.keyCode !== 27) return;
    const items = document.querySelector(".select-items");
    if (items && !items.classList.contains("hidden")) closeFactorDropdown();
  });
};

// ─────────────────────────────────────────────────────────────────────────────
// FORM RESET
// ─────────────────────────────────────────────────────────────────────────────

// Clear everything: added factors and their scores, name, age, sex and any error.
const clearAll = function() {
  document.querySelectorAll(".remove-feature").forEach(function(el){ el.click(); });
  document.getElementById("pdpForm").reset();
  const nameEl = document.getElementById("username");
  if (nameEl) nameEl.value = "";
  document.querySelectorAll('[name="gender"]').forEach(function(g){ g.checked = false; });
  removeSWSError();
  resetSelectedItemBox();
};

const resetForm = function(event) {
  if (event && event.preventDefault) event.preventDefault();
  clearAll();
};

// ─────────────────────────────────────────────────────────────────────────────
// FORM VALIDATION & SUBMIT
// ─────────────────────────────────────────────────────────────────────────────

const validateAndGetFormData = function() {
  const genderEl        = document.querySelector('[name="gender"]:checked');
  const ageEl           = document.querySelector("#Age");
  const factorsInputArr = document.querySelectorAll("input.form-control-range.factors-input");
  if (!genderEl || !genderEl.value) throw new Error("Please select the sex");
  if (!ageEl    || !ageEl.value)    throw new Error("Please enter your age");
  if (factorsInputArr.length < MIN_FEATURES)  throw new Error("Please select more than " + (MIN_FEATURES - 1) + " features to continue");
  const formData = new FormData();
  formData.append("Gender", genderEl.value.toLowerCase() === "male" ? 0 : 1);
  formData.append("Age",    ageEl.value);
  formData.append("num_features", factorsInputArr.length);
  factorsInputArr.forEach(function(item, i) {
    formData.append("Feature" + (i+1),       item.id);
    formData.append("Score_Feature" + (i+1), item.value);
  });
  return formData;
};

let _loaderTimer = null;
const LOADER_MIN_MS = 1800;  // minimum time the loader stays on screen
const showLoadingOnSubmitCTA = function() {
  // The loader appears straight away and stays for at least LOADER_MIN_MS (see the
  // submit handler), so even a fast answer gets a calm, visible "working" moment.
  clearTimeout(_loaderTimer);
  document.getElementById("fullPageLoader").classList.add("d-flex");
  const cta = document.getElementById("submitFormCTA");
  cta.innerText = "Calculating";
  cta.classList.add("disabled", "pd-busy");
};

const hideFullPageLoader = function() {
  clearTimeout(_loaderTimer);
  document.getElementById("fullPageLoader").classList.remove("d-flex");
};

const resetSubmitButton = function() {
  hideFullPageLoader();
  const cta = document.getElementById("submitFormCTA");
  cta.innerText = "Calculate my risk profile";
  cta.classList.remove("disabled", "pd-busy");
  updateSubmitReadiness();
};

$("#pdpForm").bind("submit", async function(e) {
  e.preventDefault();
  try {
    const formData = validateAndGetFormData();
    _lastFormData  = formData;
    showLoadingOnSubmitCTA();
    const t0 = performance.now();

    const resp   = await fetch("/submit", { method: "POST", body: formData });
    const result = await resp.json();
    const wait   = LOADER_MIN_MS - (performance.now() - t0);
    if (wait > 0) await new Promise(function(r) { setTimeout(r, wait); });
    if (result.error) throw new Error(result.error);

    _lastResult        = result;
    _showingFullReport = false;

    document.getElementById("resultModalLabel").innerText  = "Prediction Results";
    const uName = (document.getElementById("username") || {}).value || "";
    document.getElementById("resultModalBody").innerHTML   = getSimpleResultHTML(result, uName, result.numProvided);
    document.getElementById("resultModalFooter").innerHTML = buildModalFooter(false);
    $("#resultModal").modal();
    resetSubmitButton();   // button returns to normal as soon as the result is shown
  } catch(err) {
    resetSubmitButton();
    alert(err.message || err);
  }
});

$("#resultModal").on("hide.bs.modal", resetSubmitButton);

// ─────────────────────────────────────────────────────────────────────────────
// MISC HELPERS
// ─────────────────────────────────────────────────────────────────────────────

// The submit button stays dulled until the model's inclusion rule is met, so it is
// obvious at a glance that more factors are needed. It is deliberately still
// clickable: clicking it raises the usual "select at least N features" message.
const updateSubmitReadiness = function() {
  const cta = document.getElementById("submitFormCTA");
  if (!cta) return;
  const ready   = _factorCount >= MIN_FEATURES;
  const missing = MIN_FEATURES - _factorCount;
  cta.classList.toggle("pd-cta-idle", !ready);
  cta.setAttribute("aria-disabled", ready ? "false" : "true");
  cta.title = ready ? "" : "Add " + missing + " more factor" + (missing === 1 ? "" : "s") + " to continue";
  const hint = document.getElementById("submitHint");
  if (hint) {
    hint.textContent = ready
      ? ""
      : "At least " + missing + " more factor" + (missing === 1 ? "" : "s") + " needed";
  }
};

const updateCountOnFactorsChange = function() {
  document.querySelectorAll(".factors-count").forEach(function(el) {
    if (el.textContent !== String(_factorCount)) {
      el.textContent = _factorCount;
      el.classList.remove("pd-bump"); void el.offsetWidth; el.classList.add("pd-bump");
    }
  });
  updateSubmitReadiness();
};

const stopOnEnter = function(e) {
  if (e && (e.which === 13 || e.keyCode === 13)) return false;
};
$("#pdpForm").keypress(stopOnEnter);

const removeFeature = function(element) {
  const c = document.getElementById("container" + element.id);
  const s = document.getElementById("summary"   + element.id);
  const ev = window.event;
  if (ev && ev.isTrusted && c && !c.dataset.leaving) {
    // A real click: slide the rows out first, then remove them. Programmatic
    // clicks (Clear all, Random Fill) stay instant so the form state is exact.
    c.dataset.leaving = "1";
    $(".tooltip").hide();
    [c, s].forEach(function(el) { if (el) el.classList.add("pd-leaving"); });
    setTimeout(function() { removeFeatureNow(element); }, 180);
    return;
  }
  removeFeatureNow(element);
};

const removeFeatureNow = function(element) {
  const c = document.getElementById("container" + element.id);
  const s = document.getElementById("summary"   + element.id);
  if (c) { c.remove(); _factorCount = Math.max(0, _factorCount - 1); }
  if (s) s.remove();
  setDropdownItemDisabled(element.id, false);
  updateCountOnFactorsChange();
  $(".tooltip").hide();
};

const showInfo = function(element) {
  const item = factorConfig[element.id];
  document.getElementById("resultModalLabel").innerText  = item.label;
  document.getElementById("resultModalBody").innerHTML   = getInfoModalBody(item);
  document.getElementById("resultModalFooter").innerHTML =
    '<button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>';
  $("#resultModal").modal();
};

// Footer pages (About, Data, Disclaimer, Guide): each gets its own palette accent
// and icon. The text is wrapped in one block so inline bold text stays in the
// flow of the paragraph (the modal body is a flex column).
const INFO_PAGES = {
  about:      { color: "#36C2D3", icon: '<circle cx="8" cy="8" r="6.2"/><path d="M8 7.3v4"/><circle cx="8" cy="4.9" r=".45"/>' },
  data:       { color: "#84CDAC", icon: '<ellipse cx="8" cy="3.8" rx="5" ry="1.9"/><path d="M3 3.8v8.4c0 1 2.2 1.9 5 1.9s5-.9 5-1.9V3.8M3 8c0 1 2.2 1.9 5 1.9S13 9 13 8"/>' },
  disclaimer: { color: "#F07569", icon: '<path d="M8 1.8L14.6 13.5H1.4z"/><path d="M8 6.2v3.6"/><circle cx="8" cy="11.6" r=".45"/>' },
  guide:      { color: "#B05F9B", icon: '<path d="M2.5 3h4c.9 0 1.5.6 1.5 1.5v9c0-.8-.6-1.3-1.4-1.3H2.5zM13.5 3h-4c-.9 0-1.5.6-1.5 1.5v9c0-.8.6-1.3 1.4-1.3h4.1z"/>' },
};

const showDataModal = function(element) {
  const name = element.innerText.trim();
  const page = INFO_PAGES[name.toLowerCase()] || INFO_PAGES.about;
  document.getElementById("resultModalLabel").innerHTML =
    '<span class="pd-info-chip" style="--pd-acc:' + page.color + '">'
    + '<svg viewBox="0 0 16 16" aria-hidden="true">' + page.icon + '</svg></span>' + name;
  document.getElementById("resultModalBody").innerHTML =
    '<div class="pd-info" style="--pd-acc:' + page.color + '">' + element.dataset["info"] + '</div>';
  document.getElementById("resultModalFooter").innerHTML =
    '<button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>';
  $("#resultModal").modal();
};

const setContainerHeightObserver = function() {
  const setMaxHeight = function() {
    const rect = document.querySelector(".selected-item").getBoundingClientRect();
    document.getElementById("selectOptionsContainer").style.maxHeight =
      Math.ceil(window.innerHeight - rect.bottom - 60) + "px";
  };
  window.addEventListener("resize", setMaxHeight);
  setMaxHeight();
};
// ─────────────────────────────────────────────────────────────────────────────
// SLIDER TRACK CLICKS
// Clicking the track ahead of the handle moves the score up by one step, and
// clicking behind it moves it down by one step. Without this, a click on a
// slider with few steps (for example 0-1) snapped to the nearest value and often
// did nothing. Grabbing the handle itself still drags as usual.
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener("pointerdown", function(e) {
  const el = e.target;
  if (!el.matches || !el.matches('input[type="range"].factors-input') || e.button !== 0) return;
  const min  = Number(el.min), max = Number(el.max), step = Number(el.step) || 1;
  const val  = Number(el.value);
  if (!(max > min)) return;
  const rect   = el.getBoundingClientRect();
  const thumbW = 18;                                      // close to the rendered handle size
  const thumbX = rect.left + thumbW / 2 + (val - min) / (max - min) * (rect.width - thumbW);
  if (Math.abs(e.clientX - thumbX) <= thumbW * 0.75) return;   // on the handle: native drag
  e.preventDefault();
  const dec  = (String(step).split(".")[1] || "").length;
  const next = Math.min(max, Math.max(min, val + (e.clientX > thumbX ? step : -step)));
  el.value = next.toFixed(dec);
  el.focus({ preventScroll: true });
  handleRangeChange(el);
}, true);
