"""Structural checks for the filled A4 conference Word draft."""
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "paper" / "conference_submission.docx"


def test_filled_word_template_has_a4_two_column_body():
    doc = Document(DOCX)
    assert len(doc.sections) == 2
    for section in doc.sections:
        assert abs(section.page_width.inches - 8.27) < 0.03
        assert abs(section.page_height.inches - 11.69) < 0.03
    first_cols = doc.sections[0]._sectPr.find(qn("w:cols"))
    body_cols = doc.sections[1]._sectPr.find(qn("w:cols"))
    assert first_cols is None or first_cols.get(qn("w:num"), "1") == "1"
    assert body_cols is not None and body_cols.get(qn("w:num")) == "2"


def test_filled_word_template_uses_only_black_times_new_roman_runs():
    doc = Document(DOCX)
    runs = [r for p in doc.paragraphs for r in p.runs]
    runs += [r for t in doc.tables for row in t.rows for c in row.cells
             for p in c.paragraphs for r in p.runs]
    assert runs
    for run in runs:
        assert run.font.name == "Times New Roman"
        assert run.font.color.rgb == RGBColor(0, 0, 0)


def test_filled_word_template_contains_paper_not_template_guidance():
    doc = Document(DOCX)
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Provenance Aware Conversational" in text
    assert "Abstract." in text
    assert "Human annotation remains pending" in text
    assert "This electronic document is a" not in text
    assert "Do Not Use Symbols" not in text
    assert len(doc.tables) == 1
