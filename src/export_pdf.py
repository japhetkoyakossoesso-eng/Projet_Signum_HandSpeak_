"""
export_pdf.py — Export de l'historique en PDF pour Signum HandSpeak

Usage :
    from src.export_pdf import export_to_pdf
"""

import os
import time
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False
    print("reportlab non installé — lance : pip install reportlab")


def export_to_pdf(history: list, path: str = None) -> str:
    """
    Génère un PDF de l'historique des signes détectés.

    Args:
        history : liste des signes détectés (ex: ['A', 'B', 'bonjour'])
        path    : chemin de sortie (auto-généré si None)

    Returns:
        Chemin du fichier PDF généré
    """
    if not REPORTLAB_OK:
        print("reportlab manquant — pip install reportlab")
        return ""

    os.makedirs("exports", exist_ok=True)

    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"exports/historique_{timestamp}.pdf"

    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    #  Styles 
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        'Title',
        fontSize=24,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#00cc66'),
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    style_subtitle = ParagraphStyle(
        'Subtitle',
        fontSize=11,
        fontName='Helvetica',
        textColor=colors.HexColor('#888888'),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    style_section = ParagraphStyle(
        'Section',
        fontSize=12,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#333333'),
        spaceBefore=16,
        spaceAfter=8,
    )
    style_body = ParagraphStyle(
        'Body',
        fontSize=11,
        fontName='Helvetica',
        textColor=colors.HexColor('#222222'),
        leading=18,
    )
    style_sign = ParagraphStyle(
        'Sign',
        fontSize=18,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#0d0d1a'),
        alignment=TA_CENTER,
    )

    # Contenu 
    content = []
    now = datetime.now().strftime("%d/%m/%Y à %H:%M")

    # En-tête
    content.append(Paragraph("SIGNUM HANDSPEAK", style_title))
    content.append(Paragraph("Traducteur de Langue des Signes Française", style_subtitle))
    content.append(Paragraph(f"Rapport généré le {now}", style_subtitle))
    content.append(Spacer(1, 0.4*cm))
    content.append(HRFlowable(width="100%", thickness=1,
                               color=colors.HexColor('#00cc66')))
    content.append(Spacer(1, 0.4*cm))

    # Statistiques
    content.append(Paragraph("Statistiques", style_section))

    lettres = [s for s in history if len(s) == 1]
    mots    = [s for s in history if len(s) > 1]

    stats_data = [
        ["Métrique", "Valeur"],
        ["Total de signes détectés", str(len(history))],
        ["Lettres détectées", str(len(lettres))],
        ["Mots détectés", str(len(mots))],
        ["Signes uniques", str(len(set(history)))],
        ["Durée de session", "—"],
    ]

    stats_table = Table(stats_data, colWidths=[10*cm, 6*cm])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d0d1a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#00cc66')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#f8f8f8'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    content.append(stats_table)

    # Historique complet
    content.append(Paragraph("Historique complet", style_section))

    if history:
        texte_complet = "  ".join(history)
        content.append(Paragraph(texte_complet, style_body))
    else:
        content.append(Paragraph("Aucun signe enregistré.", style_body))

    content.append(Spacer(1, 0.4*cm))

    # Tableau des signes par ordre d'apparition
    if history:
        content.append(Paragraph("Détail des signes", style_section))

        rows = [["#", "Signe", "Type"]]
        for i, signe in enumerate(history, 1):
            type_signe = "Lettre" if len(signe) == 1 else "Mot"
            rows.append([str(i), signe.upper(), type_signe])

        # Affiche max 50 entrées
        if len(rows) > 51:
            rows = rows[:51]
            rows.append(["...", "...", "..."])

        detail_table = Table(rows, colWidths=[2*cm, 10*cm, 5*cm])
        detail_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0d0d1a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#00cc66')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.HexColor('#f8f8f8'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        content.append(detail_table)

    # Pied de page
    content.append(Spacer(1, 0.6*cm))
    content.append(HRFlowable(width="100%", thickness=0.5,
                               color=colors.HexColor('#cccccc')))
    content.append(Spacer(1, 0.2*cm))
    content.append(Paragraph(
        "Signum HandSpeak — Projet IA traducteur LSF | Japhet Koyakossoesso",
        style_subtitle
    ))

    # Génération du PDF
    doc.build(content)
    print(f"PDF généré : {path}")
    return path


if __name__ == "__main__":
    # Test avec un historique fictif
    test_history = list("BONJOUR") + ["bonjour", "merci", "aide"] + list("MERCI")
    path = export_to_pdf(test_history)
    print(f"PDF créé : {path}")
