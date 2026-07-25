"""
report_generator.py - PDF Report Generation using ReportLab
Generates professional brain tumor detection reports.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def generate_pdf_report(report_path: str, user_name: str, image_path: str,
                        result: dict, prediction_id: str, created_at: str):
    """
    Generate a professional PDF report for a brain tumor prediction.
    """
    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm
    )

    # Color palette
    PRIMARY = colors.HexColor('#1a73e8')
    DARK = colors.HexColor('#1e2340')
    LIGHT_GRAY = colors.HexColor('#f5f7fa')
    BORDER = colors.HexColor('#dde3f0')
    RED = colors.HexColor('#e53e3e')
    GREEN = colors.HexColor('#38a169')

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Title'],
        fontSize=22, textColor=PRIMARY, spaceAfter=4,
        alignment=TA_CENTER, fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle', parent=styles['Normal'],
        fontSize=11, textColor=DARK, spaceAfter=2,
        alignment=TA_CENTER, fontName='Helvetica'
    )
    section_style = ParagraphStyle(
        'SectionStyle', parent=styles['Heading2'],
        fontSize=13, textColor=PRIMARY, spaceBefore=12, spaceAfter=6,
        fontName='Helvetica-Bold', borderPad=4
    )
    body_style = ParagraphStyle(
        'BodyStyle', parent=styles['Normal'],
        fontSize=10, textColor=DARK, spaceAfter=4,
        fontName='Helvetica', leading=14
    )
    label_style = ParagraphStyle(
        'LabelStyle', parent=styles['Normal'],
        fontSize=9, textColor=colors.HexColor('#666666'),
        fontName='Helvetica-Bold'
    )
    value_style = ParagraphStyle(
        'ValueStyle', parent=styles['Normal'],
        fontSize=10, textColor=DARK, fontName='Helvetica'
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Brain Tumor Detection System", title_style))
    story.append(Paragraph("Using Deep Learning — Medical Analysis Report", subtitle_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY))
    story.append(Spacer(1, 0.15 * inch))

    # ── Report Meta Table ──────────────────────────────────────────────────────
    meta_data = [
        [Paragraph("<b>Report ID:</b>", body_style), Paragraph(prediction_id, body_style),
         Paragraph("<b>Date & Time:</b>", body_style), Paragraph(created_at, body_style)],
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(user_name, body_style),
         Paragraph("<b>Report Type:</b>", body_style), Paragraph("MRI Brain Scan Analysis", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[3.5 * cm, 7 * cm, 3.5 * cm, 7 * cm])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, BORDER),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.25 * inch))

    # ── MRI Image Section ─────────────────────────────────────────────────────
    story.append(Paragraph("MRI Scan Image", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.1 * inch))

    if os.path.exists(image_path):
        try:
            img = Image(image_path, width=3.5 * inch, height=3.5 * inch)
            img.hAlign = 'CENTER'

            # Wrap in centered table
            img_table = Table([[img]], colWidths=[4 * inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BOX', (0, 0), (-1, -1), 1, BORDER),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
            ]))
            story.append(img_table)
        except Exception:
            story.append(Paragraph("(Image could not be loaded)", body_style))
    else:
        story.append(Paragraph("(Image file not available)", body_style))

    story.append(Spacer(1, 0.25 * inch))

    # ── Prediction Results ────────────────────────────────────────────────────
    story.append(Paragraph("Prediction Results", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.1 * inch))

    result_color = RED if result.get('is_tumor') else GREEN
    result_text = result.get('display_name', 'Unknown')
    confidence = result.get('confidence', 0)

    result_data = [
        [
            Paragraph("<b>Diagnosis</b>", label_style),
            Paragraph(f"<font color='#{result_color.hexval()[2:]}' size='13'><b>{result_text}</b></font>", body_style)
        ],
        [
            Paragraph("<b>Confidence Score</b>", label_style),
            Paragraph(f"<b>{confidence}%</b>", body_style)
        ],
        [
            Paragraph("<b>Tumor Detected</b>", label_style),
            Paragraph("Yes" if result.get('is_tumor') else "No", body_style)
        ],
        [
            Paragraph("<b>Description</b>", label_style),
            Paragraph(result.get('description', ''), body_style)
        ],
    ]

    result_table = Table(result_data, colWidths=[4.5 * cm, 13 * cm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_GRAY),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.3, BORDER),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.25 * inch))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 0.1 * inch))
    disclaimer = (
        "<b>Disclaimer:</b> This report is generated by an AI-based system for "
        "educational and research purposes only. It is NOT a substitute for professional "
        "medical diagnosis. Please consult a qualified medical professional for proper "
        "diagnosis and treatment."
    )
    disc_style = ParagraphStyle(
        'Disc', parent=styles['Normal'], fontSize=8,
        textColor=colors.HexColor('#888888'), fontName='Helvetica-Oblique'
    )
    story.append(Paragraph(disclaimer, disc_style))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.15 * inch))
    footer_style = ParagraphStyle(
        'Footer', parent=styles['Normal'], fontSize=8,
        textColor=colors.HexColor('#aaaaaa'), alignment=TA_CENTER
    )
    story.append(Paragraph(
        f"Generated by Brain Tumor Detection System — {datetime.now().strftime('%Y')}",
        footer_style
    ))

    doc.build(story)
