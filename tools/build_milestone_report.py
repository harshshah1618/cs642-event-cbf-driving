"""Render the plain-language milestone as exactly two readable pages."""

import re
from pathlib import Path
from xml.sax.saxutils import escape

from driving.assets import asset_directory
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/milestone-report.md"
OUTPUT = ROOT / "reports/milestone-report.pdf"
FONTS = asset_directory(ROOT / "reports/study-evidence.zip") / "final-study/fonts"
for face, file in (
    ("MilestoneSerif", "LiberationSerif-Regular.ttf"),
    ("MilestoneSerif-Bold", "LiberationSerif-Bold.ttf"),
    ("MilestoneSerif-Italic", "LiberationSerif-Italic.ttf"),
):
    pdfmetrics.registerFont(TTFont(face, str(FONTS / file)))
pdfmetrics.registerFontFamily(
    "MilestoneSerif",
    normal="MilestoneSerif",
    bold="MilestoneSerif-Bold",
    italic="MilestoneSerif-Italic",
    boldItalic="MilestoneSerif-Bold",
)
body = ParagraphStyle("Body", fontName="MilestoneSerif", fontSize=10.6, leading=13.7, spaceAfter=8)
head = ParagraphStyle(
    "Head",
    parent=body,
    fontName="MilestoneSerif-Bold",
    fontSize=12,
    leading=15,
    spaceBefore=4,
    spaceAfter=6,
    keepWithNext=True,
)
title = ParagraphStyle("Title", parent=head, fontSize=17, leading=20, spaceAfter=7)
meta = ParagraphStyle("Meta", parent=body, fontSize=10, leading=12, spaceAfter=3)
cell = ParagraphStyle("Cell", parent=body, fontSize=9.6, leading=11.7, spaceAfter=0)
note = ParagraphStyle("Note", parent=body, fontSize=9, leading=11.3, spaceAfter=6)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#bbbbbb"))
    canvas.line(49, 38, 563, 38)
    canvas.setFont("MilestoneSerif", 9)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(49, 25, "Milestone report | November 12, 2026")
    canvas.drawRightString(563, 25, f"{doc.page} / 2")
    canvas.restoreState()


def p(text, style=body):
    return Paragraph(escape(text), style)


def main():
    lines = SOURCE.read_text().splitlines()
    story = []
    i = 0
    paragraph = []

    def flush():
        if paragraph:
            text = " ".join(paragraph)
            story.append(p(text, note if text.startswith("These comparisons use") else body))
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            flush()
            story.append(p(line[2:], title))
        elif i in (1, 2):
            story.append(p(line, meta))
            if i == 2:
                story.append(Spacer(1, 6))
        elif line.startswith("## "):
            flush()
            story.append(p(line[3:], head))
        elif line == "<!-- PAGE BREAK -->":
            flush()
            story.append(PageBreak())
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                parts = [x.strip() for x in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r"[:\-]+", v) for v in parts):
                    rows.append(parts)
                i += 1
            cells = [[p(value, cell) for value in row] for row in rows]
            t = Table(cells, colWidths=[280, 62, 58, 114], repeatRows=1, hAlign="LEFT")
            t.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c5c5c5")),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ]
                )
            )
            story += [t, Spacer(1, 7)]
            continue
        elif not line.strip():
            flush()
        else:
            paragraph.append(line)
        i += 1
    flush()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=(612, 792),
        leftMargin=49,
        rightMargin=49,
        topMargin=40,
        bottomMargin=49,
        title="Driving System Milestone Report",
        author="Akash Ravi, Harsh Shah, Ved Chadderwala",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)
    print("Words:", len(SOURCE.read_text().split()))


if __name__ == "__main__":
    main()
