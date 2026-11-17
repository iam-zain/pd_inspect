/* Top-10 clinical features driving the FINAL ensemble, by mean |SHAP value|
   (permutation SHAP over the whole pipeline, 123 original-cohort test participants,
   primary analysis with participant+visit harmonisation, retrained 2026-11-15 with the L10 collaborative-filtering M3).
   Generated from SHAP_Results/final_mean_abs_shap.csv - do not edit by hand.
   Age and sex are not selectable dropdown items and are excluded. */
window.topShapFeatures = [
  "Smell",
  "Gastrointestinal",
  "REM_MotorBehavior",
  "Trail_Making_A",
  "REM_DreamActivity",
  "Symbol_Digit_Modalities",
  "Cognitive_Category",
  "Pupillomotor",
  "Cardiovascular",
  "REM_AwakeDisturbance",
];
