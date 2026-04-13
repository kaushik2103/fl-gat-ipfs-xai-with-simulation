# ============================================================
# REPORT GENERATOR
# FL-GAT-IPFS Demo
# Generates Professional Multi-Row PDF Explanation Report
# ============================================================

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import pagesizes
from reportlab.lib.units import inch

from datetime import datetime
import tempfile
import os


# ============================================================
# GENERATE PDF REPORT (MULTI-ROW SAFE)
# ============================================================

def generate_explanation_report(
    save_path: str,
    predictions: list,
    confidences: list,
    explanations: list,
    feature_importances: list,
):
    """
    Generate a professional multi-row PDF report.

    Parameters:
    -----------
    save_path : str
        Output file name
    predictions : list[str]
    confidences : list[float]
    explanations : list[str]
    feature_importances : list[list[(feature, score)]]
    """

    # --------------------------------------------------------
    # Create temp file path
    # --------------------------------------------------------
    temp_dir = tempfile.gettempdir()
    full_path = os.path.join(
        temp_dir,
        save_path
    )

    doc = SimpleDocTemplate(
        full_path,
        pagesize=pagesizes.A4,
        rightMargin=30,
        leftMargin=30,
        topMargin=40,
        bottomMargin=30,
    )

    elements = []
    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    section_style = styles["Heading2"]
    normal_style = styles["Normal"]

    # ========================================================
    # TITLE
    # ========================================================

    elements.append(
        Paragraph("FL-GAT-IPFS Intrusion Detection Report", title_style)
    )
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(
        Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            normal_style,
        )
    )
    elements.append(Spacer(1, 0.4 * inch))

    # ========================================================
    # PREDICTION SUMMARY TABLE
    # ========================================================

    elements.append(Paragraph("Prediction Summary", section_style))
    elements.append(Spacer(1, 0.2 * inch))

    table_data = [["Row", "Prediction", "Confidence (%)"]]

    for i in range(len(predictions)):
        table_data.append([
            str(i + 1),
            str(predictions[i]),
            f"{float(confidences[i]):.2f}",
        ])

    summary_table = Table(table_data, hAlign="LEFT")

    summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (2, 1), (2, -1), "RIGHT"),
        ])
    )

    elements.append(summary_table)
    elements.append(Spacer(1, 0.5 * inch))

    # ========================================================
    # DETAILED EXPLANATIONS
    # ========================================================

    elements.append(Paragraph("Detailed Explanations", section_style))
    elements.append(Spacer(1, 0.3 * inch))

    for i in range(len(predictions)):

        elements.append(
            Paragraph(f"<b>Sample {i + 1}</b>", styles["Heading3"])
        )
        elements.append(Spacer(1, 0.1 * inch))

        elements.append(
            Paragraph(f"Prediction: {predictions[i]}", normal_style)
        )
        elements.append(
            Paragraph(f"Confidence: {float(confidences[i]):.2f}%", normal_style)
        )

        elements.append(Spacer(1, 0.1 * inch))

        explanation_text = explanations[i] if i < len(explanations) else "Explanation not available."
        elements.append(
            Paragraph(f"Explanation: {explanation_text}", normal_style)
        )

        elements.append(Spacer(1, 0.2 * inch))

        # ----------------------------------------------------
        # FEATURE IMPORTANCE TABLE (SAFE CHECK)
        # ----------------------------------------------------
        if (
            feature_importances is not None
            and i < len(feature_importances)
            and feature_importances[i] is not None
            and len(feature_importances[i]) > 0
        ):

            elements.append(
                Paragraph("Top Contributing Features:", styles["Italic"])
            )
            elements.append(Spacer(1, 0.1 * inch))

            feature_table_data = [["Feature", "Importance Score"]]

            for feature, score in feature_importances[i]:
                feature_table_data.append([
                    str(feature),
                    f"{float(score):.4f}",
                ])

            feature_table = Table(feature_table_data, hAlign="LEFT")

            feature_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ])
            )

            elements.append(feature_table)

        elements.append(Spacer(1, 0.5 * inch))

    # ========================================================
    # FOOTER
    # ========================================================

    elements.append(
        Paragraph(
            "This report was generated using FL-GAT-IPFS "
            "(Federated Graph Attention Network with FedProx, "
            "Malicious Client Detection, and IPFS-based Traceability).",
            styles["Italic"],
        )
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    doc.build(elements)

    return full_path