import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(site_data: dict, vision_data: dict, output_path: str = "outputs/site_audit_report.pdf") -> str:
    """
    Generates a PDF audit report for a telecom site.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#0E2F44'),
        spaceAfter=12
    )
    elements.append(Paragraph(f"Telecom Site Audit Report - {site_data.get('site_id', 'N/A')}", title_style))
    elements.append(Spacer(1, 10))

    # Site Performance Table
    p_data = [
        ["Metric", "Value"],
        ["Site ID", str(site_data.get("site_id", "N/A"))],
        ["Failure Risk Probability", f"{site_data.get('failure_prob', 0)*100:.1f}%"],
        ["Calculated Health Score", f"{site_data.get('health_score', 0)}%"],
        ["DG Run Hours", f"{site_data.get('dg_hours', 0)} hrs"],
        ["Battery Voltage", f"{site_data.get('battery_voltage', 0)} V"],
        ["Temperature", f"{site_data.get('temperature', 0)} °C"],
        ["Fuel Level", f"{site_data.get('fuel_level', 0)}%"],
        ["Rectifier Alarm Status", "Active" if site_data.get("rectifier_alarm") == 1 else "Normal"]
    ]
    
    t_perf = Table(p_data, colWidths=[200, 250])
    t_perf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#005F73')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING', (0,0), (-1,-1), 6)
    ]))
    elements.append(Paragraph("<b>Telemetry & Predictive Health Analytics</b>", styles['Heading2']))
    elements.append(t_perf)
    elements.append(Spacer(1, 15))

    # Vision Analysis Section
    if vision_data:
        elements.append(Paragraph("<b>Gemini AI Visual Inspection</b>", styles['Heading2']))
        v_data = [
            ["Visual Finding", "Result"],
            ["Corrosion Score", f"{vision_data.get('corrosion_score', 'N/A')}/100"],
            ["Battery Swelling", "Detected" if vision_data.get('battery_swelling') else "None"],
            ["Oil Leakage", "Detected" if vision_data.get('oil_leakage') else "None"],
            ["Missing Bolts", "Detected" if vision_data.get('missing_bolts') else "None"],
            ["Cable Damage", "Detected" if vision_data.get('cable_damage') else "None"],
            ["Housekeeping Score", f"{vision_data.get('housekeeping_score', 'N/A')}/100"],
            ["AI Summary Notes", str(vision_data.get('summary_notes', 'N/A'))]
        ]
        t_vis = Table(v_data, colWidths=[200, 250])
        t_vis.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#94D2BD')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        elements.append(t_vis)

    doc.build(elements)
    return output_path
