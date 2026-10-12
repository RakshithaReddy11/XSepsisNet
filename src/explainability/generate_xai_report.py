"""
XSepsisNet - Human Readable XAI Report Generator
=================================================

Reads the existing SHAP output and creates:

1. XSepsisNet_XAI_Report.docx
2. XSepsisNet_XAI_Feature_Importance.png

IMPORTANT:
- Does NOT retrain the BiLSTM.
- Does NOT change SHAP calculations.
- SHAP values are converted into human-readable explanations.
- Clinical suggestions are REVIEW PROMPTS, not diagnoses or treatment orders.
"""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

XAI_DIR = (
    PROJECT_ROOT
    / "models"
    / "explainability"
)

FEATURE_FILE = (
    XAI_DIR
    / "shap_feature_contributions.csv"
)

SUMMARY_FILE = (
    XAI_DIR
    / "shap_summary.json"
)

REPORT_FILE = (
    XAI_DIR
    / "XSepsisNet_XAI_Report.docx"
)

GRAPH_FILE = (
    XAI_DIR
    / "XSepsisNet_XAI_Feature_Importance.png"
)


# ============================================================
# HELPERS
# ============================================================

def set_cell_shading(cell, fill):
    """
    Set background color for a Word table cell.
    """

    tc_pr = cell._tc.get_or_add_tcPr()

    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)

    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False):
    """
    Clean table cell formatting.
    """

    cell.text = ""

    paragraph = cell.paragraphs[0]

    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(9)

    cell.vertical_alignment = (
        WD_CELL_VERTICAL_ALIGNMENT.CENTER
    )


def readable_feature(feature):
    """
    Convert model feature names into human-readable names.
    """

    names = {
        "heart_rate": "Heart Rate",
        "systolic_blood_pressure":
            "Systolic Blood Pressure",
        "diastolic_blood_pressure":
            "Diastolic Blood Pressure",
        "mean_arterial_pressure":
            "Mean Arterial Pressure",
        "respiratory_rate":
            "Respiratory Rate",
        "vital_oxygen_saturation":
            "Oxygen Saturation",
        "temperature":
            "Temperature",
        "fio2":
            "FiO₂",
        "etco2":
            "EtCO₂",
        "central_venous_pressure":
            "Central Venous Pressure",
        "pulmonary_artery_pressure_systolic":
            "Pulmonary Artery Pressure - Systolic",
        "pulmonary_artery_pressure_diastolic":
            "Pulmonary Artery Pressure - Diastolic",
        "pulmonary_artery_pressure_mean":
            "Pulmonary Artery Pressure - Mean",
        "albumin":
            "Albumin",
        "alkaline_phosphatase":
            "Alkaline Phosphatase",
        "alt":
            "ALT",
        "anion_gap":
            "Anion Gap",
        "ast":
            "AST",
        "base_excess":
            "Base Excess",
        "bicarbonate":
            "Bicarbonate",
        "bilirubin":
            "Bilirubin",
        "bun":
            "BUN",
        "calcium":
            "Calcium",
        "chloride":
            "Chloride",
        "creatinine":
            "Creatinine",
        "glucose":
            "Glucose",
        "hematocrit":
            "Hematocrit",
        "hemoglobin":
            "Hemoglobin",
        "lactate":
            "Lactate",
        "magnesium":
            "Magnesium",
        "lab_oxygen_saturation":
            "Lab Oxygen Saturation",
        "pco2":
            "pCO₂",
        "ph":
            "pH",
        "phosphate":
            "Phosphate",
        "platelets":
            "Platelets",
        "po2":
            "pO₂",
        "potassium":
            "Potassium",
        "sodium":
            "Sodium",
        "wbc":
            "WBC",
    }

    return names.get(
        feature,
        feature.replace("_", " ").title()
    )


def effect_text(shap_value):
    """
    Convert SHAP direction into plain English.
    """

    if shap_value > 0:
        return (
            "🔴 Contributed toward higher "
            "predicted sepsis risk"
        )

    if shap_value < 0:
        return (
            "🟢 Contributed in the opposite "
            "direction to the predicted risk"
        )

    return "⚪ Minimal directional contribution"


def make_review_suggestion(feature, shap_value):
    """
    Generate a safe clinical review prompt.

    These are NOT treatment instructions.
    """

    if shap_value <= 0:
        return None

    suggestions = {

        "mean_arterial_pressure":
            "Review blood-pressure and overall "
            "hemodynamic status, especially if "
            "the finding persists or worsens.",

        "systolic_blood_pressure":
            "Review the blood-pressure trend and "
            "overall hemodynamic status.",

        "diastolic_blood_pressure":
            "Review blood-pressure trends together "
            "with the overall perfusion picture.",

        "heart_rate":
            "Review heart-rate trend and assess it "
            "together with perfusion and the wider "
            "clinical picture.",

        "lactate":
            "Review lactate in clinical context and "
            "consider serial measurement when "
            "clinically appropriate.",

        "respiratory_rate":
            "Review respiratory status and the trend "
            "in respiratory rate.",

        "vital_oxygen_saturation":
            "Review oxygenation together with the "
            "patient's respiratory status.",

        "lab_oxygen_saturation":
            "Review oxygenation together with other "
            "respiratory findings.",

        "pco2":
            "Review ventilation and blood-gas findings "
            "in the overall clinical context.",

        "po2":
            "Review oxygenation and respiratory status "
            "together with other clinical findings.",

        "wbc":
            "Review WBC together with clinical evidence "
            "of infection and other laboratory findings.",

        "platelets":
            "Review platelet trend together with the "
            "overall clinical and laboratory picture.",

        "creatinine":
            "Review renal function and the trend in "
            "kidney-related laboratory findings.",

        "bun":
            "Review renal function and the wider "
            "clinical context.",

        "base_excess":
            "Review acid-base status together with "
            "other blood-gas and metabolic findings.",

        "bicarbonate":
            "Review acid-base and metabolic status "
            "with other available laboratory findings.",

        "anion_gap":
            "Review metabolic status together with "
            "other laboratory and clinical findings.",
    }

    return suggestions.get(feature)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n{FEATURE_FILE}"
        )

    df = pd.read_csv(FEATURE_FILE)

    summary = {}

    if SUMMARY_FILE.exists():

        with open(
            SUMMARY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            summary = json.load(f)

    return df, summary


# ============================================================
# CREATE GRAPH
# ============================================================

def create_graph(df):

    overall = (
        df.groupby("feature")["absolute_shap"]
        .mean()
        .sort_values(ascending=True)
        .tail(12)
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        [
            readable_feature(x)
            for x in overall.index
        ],
        overall.values
    )

    plt.xlabel(
        "Average SHAP Contribution Magnitude"
    )

    plt.ylabel(
        "Clinical Feature"
    )

    plt.title(
        "XSepsisNet – Most Influential Clinical Features"
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_FILE,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# CREATE WORD REPORT
# ============================================================

def create_report(df, summary):

    document = Document()

    # --------------------------------------------------------
    # Page setup
    # --------------------------------------------------------

    section = document.sections[0]

    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)
    section.left_margin = Inches(0.65)
    section.right_margin = Inches(0.65)

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    title = document.add_paragraph()

    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    run = title.add_run(
        "🩺 XSepsisNet\n"
        "AI Explainability Report"
    )

    run.bold = True
    run.font.size = Pt(20)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = (
        df["model_probability"]
        .dropna()
    )

    if len(probabilities) > 0:

        # Use the first explained sample.
        probability = float(
            probabilities.iloc[0]
        )

        risk_percent = probability * 100

    else:

        risk_percent = None

    p = document.add_paragraph()

    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if risk_percent is not None:

        r = p.add_run(
            f"Predicted Sepsis Risk: "
            f"{risk_percent:.2f}%"
        )

    else:

        r = p.add_run(
            "Predicted Sepsis Risk: Not available"
        )

    r.bold = True
    r.font.size = Pt(16)

    # --------------------------------------------------------
    # Explanation heading
    # --------------------------------------------------------

    heading = document.add_heading(
        "🔍 Why did the model make this prediction?",
        level=1
    )

    # --------------------------------------------------------
    # Select strongest sample
    # --------------------------------------------------------

    sample_zero = df[
        df["sample_index"] == 0
    ].copy()

    sample_zero = sample_zero.sort_values(
        "absolute_shap",
        ascending=False
    )

    # Top 10 contributors
    top_features = sample_zero.head(10)

    # --------------------------------------------------------
    # Main XAI table
    # --------------------------------------------------------

    table = document.add_table(
        rows=1,
        cols=4
    )

    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    table.style = "Table Grid"

    headers = [
        "Time",
        "Clinical Parameter",
        "Model Direction",
        "Explanation",
    ]

    for i, header in enumerate(headers):

        set_cell_text(
            table.rows[0].cells[i],
            header,
            bold=True
        )

        set_cell_shading(
            table.rows[0].cells[i],
            "D9EAF7"
        )

    for _, row in top_features.iterrows():

        feature = row["feature"]

        shap_value = float(
            row["total_shap"]
        )

        hour = int(
            row["most_influential_hour"]
        )

        cells = table.add_row().cells

        set_cell_text(
            cells[0],
            f"Hour {hour}"
        )

        set_cell_text(
            cells[1],
            readable_feature(feature)
        )

        if shap_value > 0:

            direction = "🔴 Higher risk"

        else:

            direction = "🟢 Lower / opposite direction"

        set_cell_text(
            cells[2],
            direction
        )

        set_cell_text(
            cells[3],
            effect_text(shap_value)
        )

    # --------------------------------------------------------
    # Plain English summary
    # --------------------------------------------------------

    document.add_heading(
        "💬 Simple Interpretation",
        level=1
    )

    positive = top_features[
        top_features["total_shap"] > 0
    ]

    negative = top_features[
        top_features["total_shap"] < 0
    ]

    if len(positive) > 0:

        positive_names = [
            readable_feature(x)
            for x in positive.head(5)["feature"]
        ]

        p = document.add_paragraph()

        p.add_run(
            "🔴 Factors contributing toward higher "
            "predicted risk: "
        ).bold = True

        p.add_run(
            ", ".join(positive_names)
            + "."
        )

    if len(negative) > 0:

        negative_names = [
            readable_feature(x)
            for x in negative.head(5)["feature"]
        ]

        p = document.add_paragraph()

        p.add_run(
            "🟢 Factors contributing in the opposite "
            "direction: "
        ).bold = True

        p.add_run(
            ", ".join(negative_names)
            + "."
        )

    # --------------------------------------------------------
    # Input quality
    # --------------------------------------------------------

    document.add_heading(
        "📋 Input Quality",
        level=1
    )

    p = document.add_paragraph()

    p.add_run(
        "Some clinical measurements may be unavailable "
        "for this prediction. "
    ).bold = True

    p.add_run(
        "The model explanation should therefore be "
        "interpreted together with the available "
        "clinical information."
    )

    # --------------------------------------------------------
    # Clinical review suggestions
    # --------------------------------------------------------

    document.add_heading(
        "👨‍⚕️ Clinical Review Suggestions",
        level=1
    )

    suggestions_added = set()

    for _, row in top_features.iterrows():

        feature = row["feature"]

        shap_value = float(
            row["total_shap"]
        )

        suggestion = make_review_suggestion(
            feature,
            shap_value
        )

        if (
            suggestion
            and suggestion not in suggestions_added
        ):

            p = document.add_paragraph(
                style="List Bullet"
            )

            p.add_run(
                f"{readable_feature(feature)}: "
            ).bold = True

            p.add_run(
                suggestion
            )

            suggestions_added.add(
                suggestion
            )

    # General sepsis review suggestions

    p = document.add_paragraph(
        style="List Bullet"
    )

    p.add_run(
        "Overall sepsis assessment: "
    ).bold = True

    p.add_run(
        "If sepsis is clinically suspected, "
        "follow the hospital's established sepsis "
        "screening and evaluation pathway."
    )

    p = document.add_paragraph(
        style="List Bullet"
    )

    p.add_run(
        "Reassessment: "
    ).bold = True

    p.add_run(
        "Interpret this prediction together with "
        "clinical examination, trends, laboratory "
        "results and other available information."
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    document.add_heading(
        "📊 Feature Importance",
        level=1
    )

    if GRAPH_FILE.exists():

        document.add_picture(
            str(GRAPH_FILE),
            width=Inches(6.6)
        )

        last_paragraph = (
            document.paragraphs[-1]
        )

        last_paragraph.alignment = (
            WD_ALIGN_PARAGRAPH.CENTER
        )

    # --------------------------------------------------------
    # Technical note
    # --------------------------------------------------------

    document.add_heading(
        "ℹ️ Important Note",
        level=1
    )

    note = document.add_paragraph()

    note.add_run(
        "This report explains how the trained BiLSTM "
        "used the available input features. "
    )

    note.add_run(
        "SHAP contribution direction does not by itself "
        "mean that a laboratory value or vital sign is "
        "medically abnormal. "
    )

    note.add_run(
        "Clinical interpretation requires the actual "
        "patient measurement, appropriate reference "
        "information and clinical context."
    )

    note = document.add_paragraph()

    note.add_run(
        "⚠️ This is an AI-assisted prediction and "
        "explanation, not a medical diagnosis or "
        "treatment instruction."
    ).bold = True

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    document.save(
        REPORT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("XSepsisNet - Human Readable XAI Report")
    print("=" * 70)

    print("\nLoading SHAP results...")

    df, summary = load_data()

    print(
        f"Loaded {len(df)} SHAP contribution rows."
    )

    print("\nCreating readable XAI graph...")

    create_graph(df)

    print(
        f"Created:\n{GRAPH_FILE}"
    )

    print("\nCreating Word report...")

    create_report(
        df,
        summary
    )

    print(
        f"Created:\n{REPORT_FILE}"
    )

    print("\n" + "=" * 70)
    print("XAI REPORT COMPLETE")
    print("=" * 70)

    print("\nOpen the report with:")

    print(
        f'explorer "{XAI_DIR}"'
    )


if __name__ == "__main__":
    main()