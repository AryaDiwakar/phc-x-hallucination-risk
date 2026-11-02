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


def test_results_tables_stay_within_one_ieee_column():
    doc = Document(DOCX)
    for table in doc.tables[1:]:
        width = table._tbl.tblPr.find(qn("w:tblW"))
        assert width is not None
        assert int(width.get(qn("w:w"))) <= 4810


def test_paper_title_and_headings_are_bold_uppercase():
    doc = Document(DOCX)
    title = doc.paragraphs[0]
    assert title.text == title.text.upper()
    assert all(run.bold for run in title.runs)
    assert all(run.font.size.pt == 20 for run in title.runs)
    headings = [p for p in doc.paragraphs if p.style.name.startswith("Heading")]
    assert headings
    for paragraph in headings:
        assert paragraph.text == paragraph.text.upper()
        assert all(run.bold for run in paragraph.runs)
    body = next(p for p in doc.paragraphs if p.style.name == "Body Text")
    assert not any(run.bold for run in body.runs)


def test_filled_word_template_contains_paper_not_template_guidance():
    doc = Document(DOCX)
    text = "\n".join(p.text for p in doc.paragraphs)
    author_text = "\n".join(c.text for t in doc.tables[:1] for c in t.rows[0].cells)
    assert "PHC-X: PROVENANCE-AWARE" in text
    assert "Abstract—" in text
    assert "no human annotations have been collected" in text
    assert "23BDS0023" in author_text and "23BDS0212" in author_text
    assert "23BDS0019" in author_text
    assert author_text.count("School of Computer Science\nand Engineering") == 3
    assert "VIT Vellore" in author_text
    assert "This electronic document is a" not in text
    assert "Do Not Use Symbols" not in text
    assert len(doc.tables) == 7  # author grid plus six compact, one-column tables
