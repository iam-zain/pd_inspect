"""
pdf_report.py — PD-INSPECT PDF report generator
Uses reportlab to produce a single A4 page academic-style risk report.
Call generate_report(data) → returns bytes of the PDF.
"""

import re
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

# ── Colour palette ────────────────────────────────────────────────────────────
TEAL        = colors.HexColor("#2E3A59")
TEAL_LIGHT  = colors.HexColor("#F1EFFB")
RED         = colors.HexColor("#F07569")
RED_LIGHT   = colors.HexColor("#FDE7E4")
GREEN       = colors.HexColor("#4FA77F")   # mint, deepened for text contrast
GREEN_LIGHT = colors.HexColor("#E6F5EE")
ORANGE      = colors.HexColor("#E3A687")
GREY        = colors.HexColor("#95a5a6")
GREY_LIGHT  = colors.HexColor("#f8f9fa")
WHITE       = colors.white
BLACK       = colors.HexColor("#2E3A59")

# ── Style helpers ─────────────────────────────────────────────────────────────

def _style(name, **kwargs):
    defaults = dict(fontName="Helvetica", fontSize=10, textColor=BLACK, leading=14)
    defaults.update(kwargs)
    return ParagraphStyle(name, **defaults)

H1    = _style("H1", fontName="Helvetica-Bold", fontSize=16, textColor=TEAL,    alignment=TA_CENTER, leading=20)
H2    = _style("H2", fontName="Helvetica-Bold", fontSize=11, textColor=TEAL,    leading=16)
BODY  = _style("BODY", fontSize=9,  leading=13)
SMALL = _style("SMALL", fontSize=8, textColor=GREY, leading=11)
DISC  = _style("DISC", fontSize=7.5, textColor=GREY, leading=10, alignment=TA_JUSTIFY)
BOLD  = _style("BOLD", fontName="Helvetica-Bold", fontSize=9, leading=13)
CENTER= _style("CENTER", fontSize=9, leading=13, alignment=TA_CENTER)
RIGHT = _style("RIGHT", fontSize=9, leading=13, alignment=TA_RIGHT)

RISK_HIGH = _style("RISK_HIGH", fontName="Helvetica-Bold", fontSize=18,
                   textColor=RED, alignment=TA_CENTER, leading=24)
RISK_LOW  = _style("RISK_LOW",  fontName="Helvetica-Bold", fontSize=18,
                   textColor=GREEN, alignment=TA_CENTER, leading=24)

# ── Probability bar (table-based, no drawing canvas needed) ──────────────────

def _prob_bar_table(label, prob, bar_color, width_mm=60):
    """Return a 1-row Table that looks like a progress bar."""
    pct     = round(prob * 100)
    filled  = max(1, round(width_mm * prob))
    empty   = width_mm - filled

    bar_data = [[""]]
    bar_style = [
        ("BACKGROUND", (0,0), (0,0), bar_color),
        ("ROWHEIGHT",  (0,0), (0,0), 5),
        ("LEFTPADDING",(0,0), (0,0), 0),("RIGHTPADDING",(0,0),(0,0),0),
        ("TOPPADDING", (0,0), (0,0), 0),("BOTTOMPADDING",(0,0),(0,0),0),
    ]
    bar_inner = Table(bar_data, colWidths=[filled*mm], rowHeights=[5*mm])
    bar_inner.setStyle(TableStyle(bar_style))

    # outer track
    outer_data = [[bar_inner, ""]]
    outer = Table(outer_data, colWidths=[filled*mm, empty*mm], rowHeights=[5*mm])
    outer.setStyle(TableStyle([
        ("BACKGROUND", (1,0),(1,0), colors.HexColor("#e8e8e8")),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING", (0,0),(-1,-1),0),("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))

    row = Table(
        [[Paragraph(label, BODY), Paragraph(str(pct)+"%", BOLD), outer]],
        colWidths=[55*mm, 12*mm, width_mm*mm]
    )
    row.setStyle(TableStyle([
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),2),
        ("TOPPADDING", (0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2),
    ]))
    return row

# ── Main generator ────────────────────────────────────────────────────────────

def _render(data: dict, S: float = 1.0):
    """
    Render the report at scale S (1.0 = roomy, lower = tighter) and return
    (pdf_bytes, page_count). generate_report() picks the largest S that fits one page.

    data keys:
        patientName  str
        age          str/int
        gender       str  ("Male"/"Female")
        features     list of {label, score}
        result       dict from /submit JSON
    Returns bytes of a single-page A4 PDF.
    """
    buf = BytesIO()

    W, H = A4
    margin = 18*mm * (0.75 + 0.25*S)

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=16*mm*S, bottomMargin=12*mm*S,
    )

    r       = data.get("result", {})
    isPD    = r.get("userStatus") == "PROFILE_1"
    n_provided  = r.get("numProvided", len(data.get("features", [])))
    n_stacker   = len(r.get("usedFeatures", [])) + len(r.get("defaultedFeatures", []))
    n_defaulted = max(0, n_stacker - n_provided)
    risk_lbl = "HIGH RISK — Parkinson's Disease" if isPD else "LOW RISK — Healthy / No PD"
    risk_sty = RISK_HIGH if isPD else RISK_LOW
    risk_bg  = RED_LIGHT  if isPD else GREEN_LIGHT
    risk_col = RED        if isPD else GREEN

    band      = r.get("confidenceBand", "")
    band_col  = RED if "Low" in band else (ORANGE if "Moderate" in band else GREEN)
    conf_pct  = r.get("confidenceScore", 0)
    final_prob = r.get("finalProb", 0)

    story = []

    # ── HEADER ────────────────────────────────────────────────────────────────
    header_data = [[
        Paragraph("PD-INSPECT", _style("HH", fontName="Helvetica-Bold", fontSize=20,
                                        textColor=WHITE, alignment=TA_LEFT, leading=24)),
        Paragraph("Parkinson&#39;s Disease- Integrated Non-motor Symptom-based<br/>"
                  "Prediction and Evaluation using Computational Tool",
                  _style("HS", fontSize=8, textColor=colors.HexColor("#DCD6F7"),
                         alignment=TA_RIGHT, leading=11)),
    ]]
    hdr = Table(header_data, colWidths=[80*mm, W - 2*margin - 80*mm])
    hdr.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), TEAL),
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0),(-1,-1), 8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(0,0),10),("RIGHTPADDING",(-1,-1),(-1,-1),10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 5*S*mm))

    # ── DATE + PATIENT INFO ───────────────────────────────────────────────────
    name   = data.get("patientName") or "User"
    age    = data.get("age", "—")
    gender = data.get("gender", "—")
    date   = datetime.now().strftime("%d %B %Y")
    # Sex in the same colours as the web form: cyan for male, plum for female
    # (cyan deepened slightly so it reads on white).
    sex_col  = {"male": "#1FA5B6", "female": "#B05F9B"}.get(str(gender).strip().lower())
    sex_html = ('<font color="%s"><b>%s</b></font>' % (sex_col, gender)) if sex_col else str(gender)

    info_data = [
        [Paragraph("<b>Patient Name</b>", BODY), Paragraph(str(name),   BODY),
         Paragraph("<b>Date</b>",         BODY), Paragraph(date,         BODY)],
        [Paragraph("<b>Age</b>",          BODY), Paragraph(str(age)+" yrs", BODY),
         Paragraph("<b>Sex</b>",          BODY), Paragraph(sex_html,     BODY)],
    ]
    col_w = (W - 2*margin) / 4
    info_tbl = Table(info_data, colWidths=[col_w*0.55, col_w*1.15, col_w*0.55, col_w*1.15])
    info_tbl.setStyle(TableStyle([
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0),(-1,-1), GREY_LIGHT),
        ("TOPPADDING", (0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("GRID", (0,0),(-1,-1), 0.3, colors.HexColor("#dce1e3")),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 4*S*mm))

    # ── RISK OUTCOME BOX ─────────────────────────────────────────────────────
    pd_pct  = final_prob * 100
    hl_pct  = 100 - pd_pct
    primary_pct = pd_pct if isPD else hl_pct
    risk_tbl = Table(
        [[Paragraph(risk_lbl, risk_sty),
          Paragraph(
            "<b>%.1f%%</b>" % primary_pct,
            _style("PP", fontName="Helvetica-Bold", fontSize=18,
                   textColor=risk_col, alignment=TA_CENTER, leading=22))]],
        colWidths=[(W-2*margin)*0.74, (W-2*margin)*0.26]
    )
    risk_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), risk_bg),
        ("TOPPADDING", (0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("VALIGN",     (0,0),(-1,-1),"MIDDLE"),
        ("LINEBELOW",  (0,0),(-1,-1), 1.5, risk_col),
    ]))
    story.append(risk_tbl)
    story.append(Spacer(1, 4*S*mm))

    # ── TWO-COLUMN SECTION: Features | Confidence + Models ───────────────────
    left_story  = []
    right_story = []

    # LEFT — Features entered
    left_story.append(Paragraph("Clinical Features Provided", H2))
    left_story.append(Spacer(1, 1.5*S*mm))
    feats = data.get("features", [])
    if feats:
        # Two feature/score pairs per printed row, so 30 features need 15 rows, not 30.
        n_feats = len(feats)
        fs   = (8.2 if n_feats <= 12 else 7.4 if n_feats <= 20 else 6.9) * (0.85 + 0.15*S)
        lead = fs + 2.2
        pad  = 3 if n_feats <= 12 else 2 if n_feats <= 20 else 1

        def _short(f):
            lbl = re.sub(r"^Score of ", "", f.get("label", f.get("name", "")))
            if n_feats > 20:                     # drop the instrument tag to keep the type readable
                lbl = re.sub(r"\s*\[[^]]*\]", "", lbl)
            return lbl

        cell  = lambda t, c=False: Paragraph(t, _style("FV", fontSize=fs, leading=lead,
                                                       alignment=TA_CENTER if c else TA_LEFT))
        head  = lambda t, c=False: Paragraph("<b>%s</b>" % t,
                                             _style("FH", fontSize=fs, fontName="Helvetica-Bold",
                                                    alignment=TA_CENTER if c else TA_LEFT))
        half      = (n_feats + 1) // 2
        feat_rows = [[head("Feature"), head("Score", True), head("Feature"), head("Score", True)]]
        for i in range(half):
            a = feats[i]
            b = feats[i + half] if i + half < n_feats else None
            feat_rows.append([
                cell(_short(a)), cell(str(a.get("score", "")), True),
                cell(_short(b)) if b else "", cell(str(b.get("score", "")), True) if b else "",
            ])
        lw = (W - 2*margin) * 0.46
        ft = Table(feat_rows, colWidths=[lw*0.33, lw*0.17, lw*0.33, lw*0.17])
        ft.setStyle(TableStyle([
            ("BACKGROUND",   (0,0),(-1,0),  TEAL),
            ("TEXTCOLOR",    (0,0),(-1,0),  WHITE),
            ("BACKGROUND",   (0,1),(-1,-1), GREY_LIGHT),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, GREY_LIGHT]),
            ("GRID",         (0,0),(-1,-1), 0.3, colors.HexColor("#dce1e3")),
            ("TOPPADDING",   (0,0),(-1,-1), pad),("BOTTOMPADDING",(0,0),(-1,-1),pad),
            ("LEFTPADDING",  (0,0),(-1,-1), 3),("RIGHTPADDING",(0,0),(-1,-1),3),
            ("VALIGN",       (0,0),(-1,-1),"MIDDLE"),
        ]))
        left_story.append(ft)
        feat_note = str(n_provided) + " feature(s) measured; " + str(n_defaulted) + " filled with class-balanced defaults."
        left_story.append(Spacer(1, 2*S*mm))
        left_story.append(Paragraph(feat_note, SMALL))
    else:
        left_story.append(Paragraph("No features recorded.", BODY))

    # RIGHT — Confidence + sub-model votes + probabilities
    # Confidence
    right_story.append(Paragraph("Assessment Confidence", H2))
    right_story.append(Spacer(1, 1.5*S*mm))

    conf_data = [
        [Paragraph("Confidence Band", BODY),
         Paragraph("<b>%s</b>" % band,
                   _style("CB", fontName="Helvetica-Bold", fontSize=9, textColor=band_col))],
        [Paragraph("Confidence Score", BODY),
         Paragraph("<b>%d%%</b>" % conf_pct,
                   _style("CS", fontName="Helvetica-Bold", fontSize=9))],
        [Paragraph("Probability Margin", BODY),
         Paragraph("%.1f%%" % (r.get("probabilityMargin",0)*100), BODY)],
        [Paragraph("Model Consensus", BODY),
         Paragraph("%.0f%%" % (r.get("modelConsensus",0)*100), BODY)],
    ]
    rw = (W - 2*margin) * 0.52
    ct = Table(conf_data, colWidths=[rw*0.56, rw*0.44])
    ct.setStyle(TableStyle([
        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
        ("BACKGROUND",  (0,0),(-1,-1), GREY_LIGHT),
        ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#dce1e3")),
        ("TOPPADDING",  (0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING", (0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
    ]))
    right_story.append(ct)
    right_story.append(Spacer(1, 3*S*mm))

    # Sub-model votes
    right_story.append(Paragraph("Sub-Model Votes", H2))
    right_story.append(Spacer(1, 1.5*S*mm))
    votes = r.get("subModelVotes", {})
    vote_rows = []
    for model, vote in votes.items():
        v_col = RED if vote == "PD" else GREEN
        short = model.replace("M1 (Symptom-wise)","M1 Symptom-wise")\
                     .replace("M2 (Joint-symptom)","M2 Joint-symptom")\
                     .replace("M3 (Participant-profile)","M3 Participant-profile")
        vote_rows.append([
            Paragraph(short, _style("VR", fontSize=8.5, leading=12)),
            Paragraph("<b>" + vote + "</b>",
                      _style("VV", fontName="Helvetica-Bold", fontSize=8.5,
                             textColor=v_col, alignment=TA_CENTER, leading=12)),
        ])
    if not vote_rows:      # no sub-model votes in the payload - keep the PDF renderable
        vote_rows = [[Paragraph("Not available", _style("VR", fontSize=8.5, leading=12)),
                      Paragraph("-", _style("VV", fontSize=8.5, alignment=TA_CENTER, leading=12))]]
    vt = Table(vote_rows, colWidths=[rw*0.72, rw*0.28])
    vt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), GREY_LIGHT),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#dce1e3")),
        ("TOPPADDING",    (0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
        ("LEFTPADDING",   (0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
        ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
    ]))
    right_story.append(vt)

    # Assemble two-column layout
    left_cell  = [item for item in left_story]
    right_cell = [item for item in right_story]

    two_col = Table(
        [[left_cell, right_cell]],
        colWidths=[(W-2*margin)*0.50, (W-2*margin)*0.50]
    )
    two_col.setStyle(TableStyle([
        ("VALIGN",       (0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",  (0,0),(0,0),0),
        ("RIGHTPADDING", (0,0),(0,0),4*mm),
        ("LEFTPADDING",  (1,0),(1,0),0),
        ("RIGHTPADDING", (1,0),(1,0),0),
        ("TOPPADDING",   (0,0),(-1,-1),0),
        ("BOTTOMPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 4*S*mm))

    # ── PER-MODEL PROBABILITY BARS ────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=TEAL_LIGHT))
    story.append(Spacer(1, 2*S*mm))
    story.append(Paragraph("Per-Model Probability Breakdown", H2))
    story.append(Spacer(1, 2*S*mm))

    bar_defs = [
        ("M1  Symptom-wise",  r.get("probM1",0),  colors.HexColor("#36C2D3")),
        ("M2  Joint-symptom",     r.get("probM2",0),  colors.HexColor("#B05F9B")),
        ("M3  Participant-profile",  r.get("probM3",0),  colors.HexColor("#84CDAC")),
        ("Final Ensemble", r.get("finalProb",0), risk_col),
    ]

    bar_rows = []
    for label, prob, color in bar_defs:
        bar_rows.append([_prob_bar_table(label, prob, color, width_mm=62)])

    bt = Table(bar_rows, colWidths=[W - 2*margin])
    bt.setStyle(TableStyle([
        ("TOPPADDING",   (0,0),(-1,-1),1),("BOTTOMPADDING",(0,0),(-1,-1),1),
        ("LEFTPADDING",  (0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
    ]))
    story.append(bt)
    story.append(Spacer(1, 3*S*mm))

    # ── METHODOLOGY NOTE ─────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=TEAL_LIGHT))
    story.append(Spacer(1, 2*S*mm))
    method = ("Confidence = 0.70 x Probability Margin + 0.30 x Model Consensus.  "
              "M1: symptom-wise ensemble (LR + SVM + XGB per item, XGBoost stacker).  "
              "M2: joint-symptom XGBoost model (n=400, lr=0.05, d=4).  "
              "M3: collaborative filtering - visit-averaged, feature-weighted profile matched to the 80 most similar training participants (kernel-weighted votes).  "
              "Final: Logistic-regression meta-stacker on M1/M2/M3 out-of-fold predictions; "
              "30 non-motor features. Probability Margin is the distance from the decision "
              "threshold, which is fixed on training data (Youden J) and never tuned on the test set.")
    story.append(Paragraph(method, SMALL))
    story.append(Spacer(1, 2*S*mm))

    # ── DISCLAIMER ───────────────────────────────────────────────────────────
    disc_tbl = Table(
        [[Paragraph(
            "DISCLAIMER: PD-INSPECT is an academic research tool. The project is developed at Dr. Pankaj Singh Dholaniya's laboratory from Department of Biotechnology &amp; Bioinformatics at School of Life Sciences, University of Hyderabad, India. This work is a part of doctoral thesis of Mr. Md Zainul Ali. "
            "It does NOT constitute a clinical diagnosis and must not replace professional medical advice. "
            "Consult a qualified neurologist for evaluation. Results subject to inherent model limitations.",
            DISC
        )]],
        colWidths=[W - 2*margin]
    )
    disc_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(-1,-1), TEAL_LIGHT),
        ("TOPPADDING",  (0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING", (0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
        ("LINEABOVE",   (0,0),(-1,-1), 1, TEAL),
    ]))
    story.append(disc_tbl)

    doc.build(story)
    return buf.getvalue(), doc.page


def generate_report(data: dict) -> bytes:
    """
    Build the report and guarantee it fits on ONE A4 page.

    The layout is rendered at decreasing scales until the page count is 1, so a report
    with 30 features is simply tighter than one with 16 rather than spilling over.
    """
    pdf = None
    for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        pdf, pages = _render(data, scale)
        if pages <= 1:
            return pdf
    return pdf
