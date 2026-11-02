"""Fill a copy of the supplied IEEE A4 Word template with the paper.

The provided DOCX uses Strict OOXML namespace URIs, which python-docx does not
read directly. This script normalizes a temporary copy for editing, leaves the
original template untouched, and writes a separate, two-column submission
draft at ``paper/conference_submission.docx``. All text runs are explicitly
Times New Roman and black.

The prose is an editorial draft for the authors to review. This script does
not claim that the text was human-authored; authors remain responsible for
their conference's AI-use and authorship policies.
"""
from __future__ import annotations

import re
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "conference-template-a4.docx"
OUT = ROOT / "paper" / "conference_submission.docx"
MAIN_TEX = ROOT / "paper" / "main.tex"

# This file is Strict OOXML; normalize only a temporary copy so the original
# Word template supplied by the user is preserved byte-for-byte.
STRICT_TO_TRANSITIONAL = {
    "http://purl.oclc.org/ooxml/wordprocessingml/main":
        "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://purl.oclc.org/ooxml/drawingml/main":
        "http://schemas.openxmlformats.org/drawingml/2006/main",
    "http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing":
        "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "http://purl.oclc.org/ooxml/officeDocument/relationships/":
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/",
    "http://purl.oclc.org/ooxml/officeDocument/relationships":
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "http://purl.oclc.org/ooxml/officeDocument/customXml":
        "http://schemas.openxmlformats.org/officeDocument/2006/customXml",
    "http://purl.oclc.org/ooxml/officeDocument/docPropsVTypes":
        "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes",
    "http://purl.oclc.org/ooxml/officeDocument/extendedProperties":
        "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties",
    "http://purl.oclc.org/ooxml/officeDocument/math":
        "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def _normalized_template(path: Path) -> Path:
    tmp = Path(tempfile.gettempdir()) / "phcx-conference-template-transitional.docx"
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename.endswith((".xml", ".rels")):
                text = data.decode("utf-8", "replace")
                for old, new in sorted(STRICT_TO_TRANSITIONAL.items(),
                                       key=lambda item: -len(item[0])):
                    text = text.replace(old, new)
                data = text.encode("utf-8")
            dst.writestr(info, data)
    return tmp


def _set_run(run, size: float | None = None, bold: bool | None = None,
             italic: bool | None = None) -> None:
    run.font.name = "Times New Roman"
    run.font.color.rgb = RGBColor(0, 0, 0)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{attr}"), "Times New Roman")


def _set_style(style, size: float, bold: bool = False, italic: bool = False) -> None:
    style.font.name = "Times New Roman"
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = italic
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style._element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{attr}"), "Times New Roman")


def _clear_body(document) -> None:
    body = document._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def _add_text(document, text: str, size: float = 10, bold: bool = False,
              italic: bool = False, align=None, style: str | None = None,
              after: float = 2):
    p = document.add_paragraph(style=style) if style else document.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(after)
    run = p.add_run(text)
    _set_run(run, size=size, bold=bold, italic=italic)
    return p


def _add_heading(document, text: str, level: int = 1):
    if level == 1:
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(7)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(text.upper())
        _set_run(r, size=10, bold=True)
    else:
        p = document.add_paragraph()
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(text)
        _set_run(r, size=10, bold=True, italic=True)
    return p


def _add_body(document, text: str):
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    _set_run(r, size=9.5)
    return p


def _cell_text(cell, text: str, bold: bool = False):
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(str(text))
    _set_run(r, size=8, bold=bold)


def _add_table(document, headers: List[str], rows: List[Tuple[str, ...]],
               caption: str):
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(3)
    _set_run(p.add_run(caption), size=8, bold=True)
    table = document.add_table(rows=1, cols=len(headers))
    table.autofit = True
    for cell, text in zip(table.rows[0].cells, headers):
        _cell_text(cell, text, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            _cell_text(cell, text)
    document.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def _read_references() -> List[str]:
    text = MAIN_TEX.read_text(encoding="utf-8")
    block = re.search(r"\\begin\{thebibliography\}.*?(.*?)\\end\{thebibliography\}",
                      text, re.S).group(1)
    entries = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{[^}]+\}(.*?)(?=\\bibitem|$)",
                         block, re.S)
    refs = []
    for entry in entries:
        value = entry.replace("\n", " ")
        value = re.sub(r"\\[A-Za-z]+", "", value)
        value = value.replace("{", "").replace("}", "").replace("~", " ")
        value = re.sub(r"\s+", " ", value).strip()
        refs.append(value)
    return refs


def build() -> Path:
    template_copy = _normalized_template(TEMPLATE)
    doc = Document(template_copy)
    _clear_body(doc)

    # Use the supplied template's A4 page dimensions and its IEEE text margins.
    for section in doc.sections:
        section.page_width = Inches(8.27)
        section.page_height = Inches(11.69)
        section.left_margin = Inches(0.625)
        section.right_margin = Inches(0.625)
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(1.0)
        section.header_distance = Inches(0.5)
        section.footer_distance = Inches(0.5)

    # Explicit Times New Roman and black across the template's base styles.
    for name, size, bold, italic in (
        ("Normal", 9.5, False, False),
        ("paper title", 24, True, False),
        ("Author", 10, False, False),
        ("Abstract", 9, False, False),
        ("Keywords", 9, False, True),
        ("Heading 1", 10, True, False),
        ("Heading 2", 9.5, True, True),
        ("Heading 5", 9, True, False),
    ):
        try:
            _set_style(doc.styles[name], size, bold=bold, italic=italic)
        except KeyError:
            pass

    # Front matter occupies one full-width section.
    sec1 = doc.sections[0]
    cols1 = sec1._sectPr.find(qn("w:cols"))
    if cols1 is not None:
        cols1.set(qn("w:num"), "1")

    title = _add_text(
        doc,
        "Provenance Aware Conversational Pre Generation Hallucination Risk Prediction and Intervention",
        size=21, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=7,
    )
    title.paragraph_format.keep_with_next = True
    _add_text(doc, "Arya D     Jeswanth G     N Sai Baba", size=10.5,
              align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
    _add_text(doc, "Capstone Project", size=9.5,
              align=WD_ALIGN_PARAGRAPH.CENTER, after=1)
    _add_text(doc, "Affiliation and author contact details to be completed",
              size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=7)

    # Switch to the template's two-column body after the author block.
    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    sec.page_width = Inches(8.27)
    sec.page_height = Inches(11.69)
    sec.left_margin = Inches(0.625)
    sec.right_margin = Inches(0.625)
    sec.top_margin = Inches(0.75)
    sec.bottom_margin = Inches(1.0)
    cols = sec._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sec._sectPr.append(cols)
    cols.set(qn("w:num"), "2")
    cols.set(qn("w:space"), "360")

    # Abstract and keywords, written as plain text under the template's
    # component heads; no math/symbols/footnotes appear in the abstract.
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    _set_run(p.add_run("Abstract. "), 9, bold=True)
    _set_run(p.add_run(
        "We study whether an assistant's next reply can be screened before it is written. "
        "PHC X combines conversation history, claim provenance, consistency checks, and "
        "evidence coverage to estimate risk and recommend an intervention. On a synthetic "
        "benchmark with a shared prompt pool, selected XGBoost reaches an AUROC of 0.975 "
        "on 107 held out prompts; the prompt only ablation is 0.434. The model was selected "
        "using family grouped cross validation on training data, with the test split held "
        "out. On the original benchmark, a prompt template lookup baseline reaches 0.808, "
        "so we report both benchmark versions. A context coverage counterfactual derived "
        "from HaluEval reaches an AUROC of 0.705. Human annotation remains pending."
    ), 9)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    _set_run(p.add_run("Index Terms: "), 9, bold=True, italic=True)
    _set_run(p.add_run("hallucination risk, conversational systems, provenance, calibration, intervention"), 9, italic=True)

    sections = [
        ("I. Introduction", [
            "A factual error in a chat system is usually detected only after the system has produced an answer. Self consistency methods compare several generated responses, while verification methods generate a draft and then check it [1], [2], [5], [6]. These approaches can be useful, but they cannot prevent the first unsupported answer from reaching the user.",
            "Pre generation methods move the decision earlier, but most published probes inspect a model's internal states for one prompt at a time [7]–[13]. That leaves out an important source of risk in a multi turn conversation: an assistant may rely on its own earlier unsupported statement, change which dataset or model is under discussion, or answer a follow up whose premise was never established.",
            "PHC X addresses this setting. Before a reply is generated, it builds a provenance aware view of the conversation, checks for several kinds of cross turn inconsistency, estimates whether available evidence covers the question, and maps the resulting features to a calibrated risk score and an intervention."
        ]),
        ("II. Related Work", [
            "Post generation detection includes sampling based consistency checks, semantic uncertainty measures, and generate then verify pipelines [1]–[6]. Pre generation work uses hidden state or attention signals to estimate whether a single prompt will elicit an unsupported answer [7]–[13]. Dialogue benchmarks such as DiaHalu and HalluHard evaluate errors after a response exists [14], [15]; conversational guardrails instead track session risk or constrain memory writes [16]–[19]. PHC X differs in its target: risk in the next turn, estimated from conversation state before that turn is generated."
        ]),
        ("III. Method", [
            "The system converts the conversation and pending prompt into a fixed feature vector. It uses deterministic text processing and does not require access to model logits or hidden states in its standard operating mode."
        ]),
        ("A. Provenance aware memory", [
            "Conversation turns are split into candidate claims. Each claim records its speaker, source, and verification status. Source verified and retrieved claims receive higher trust than user assertions; an unverified assistant claim receives very low trust. The model therefore cannot make a claim trustworthy simply by repeating it in a later turn.",
            "The contamination score measures whether the pending prompt depends on an unverified claim previously produced by the assistant. In a controlled example, the question ‘How many samples does Atlas A contain?’ is low risk when the earlier assistant turn is marked as verified evidence, and substantially higher when the identical statement has no source."
        ]),
        ("B. Consistency checks", [
            "Five rule based engines identify cross turn contradiction, entity drift, temporal staleness, unsupported presuppositions, and ambiguous references. For example, a question asking why a score improved from one value to another presupposes that the change occurred. An entity tracker detects when the conversation silently switches datasets or models. A temporal engine raises uncertainty when a user asks for current information but the relevant memory contains only old or undated claims.",
            "These rules are deliberately transparent. They provide bounded scores and short reasons that can be shown to a user, rather than treating the classifier as an unexplained judge."
        ]),
        ("C. Evidence and risk model", [
            "A local sentence retriever uses deterministic hashed n gram similarity to rank passages. The evidence engine estimates coverage, contradiction, staleness, and whether the apparent support comes only from model generated claims. The feature engine combines prompt, history, provenance, consistency, evidence, and optional white box signals. The default model uses the black box feature groups.",
            "A random forest is used for the standard benchmark. For the leak controlled benchmark, a five fold family grouped sweep on the training split selected XGBoost with depth four. In each case calibration and intervention thresholds use validation data; the test split is reserved for final evaluation."
        ]),
        ("D. Intervention policy", [
            "The policy maps risk to five actions: generate, generate with evidence, retrieve, clarify, or abstain. At high risk it prefers retrieval or clarification; the browser extension requires an explicit user override for blocking actions and fails open if the local service is unavailable. Threshold selection includes a minimum coverage constraint so the policy does not block nearly every turn."
        ]),
        ("IV. Evaluation", [
            "PHC Conversation is a synthetic benchmark with 1,100 prompts from 238 conversation families and eleven categories. Family splitting keeps variants of the same source question in one partition. Labels combine a construction based label and a rule verified label; their agreement is reported rather than treated as perfect.",
            "The standard benchmark contains a prompt template leak: category specific wording allows a lookup table that never reads the conversation to reach an AUROC of 0.808. We therefore built a leak controlled variant with one shared prompt pool and seven context borne categories. The lookup baseline falls to 0.635, and the prompt only model is at chance. This variant is the primary measure of conversation based discrimination.",
            "The model was selected and calibrated without using the test split. Metrics include AUROC, area under the precision recall curve, F1, Brier score, and ten bin expected calibration error. We also evaluate action level safety and run cross label source and leave one category out stress tests."
        ]),
        ("V. Results", [
            "On the leak controlled test split, XGBoost reaches an AUROC of 0.975 and an area under the precision recall curve of 0.988. Across three seeds, AUROC is 0.9753 with a standard deviation of 0.0010. The action policy passes its gates at 50.5 percent coverage and catches 98.6 percent of positive cases. On the standard, prompt leaky benchmark, the random forest reaches AUROC 0.987; that higher number is reported alongside the lookup baseline, not as the primary result.",
            "The leak controlled ablation shows where the signal comes from. Prompt only is 0.434, history and prompt are 0.749, adding provenance reaches 0.779, contradiction reaches 0.783, and adding evidence reaches 0.975. The result supports the claim that the conversation and evidence, rather than question wording alone, carry most of the predictive signal.",
            "Training on the construction based label and evaluating against the verifier label gives AUROC 0.779. The reverse direction gives 0.671. In leave one category out evaluation, the mean AUROC is 0.804 across nine categories, with substantial variation between categories."
        ]),
        ("VI. External Evaluation", [
            "External results depend on what the labels ask the system to predict. DiaHalu labels a response after it has been generated; after aligning each label to its annotated turn and stratifying by history length, PHC X is near chance. The HaluEval correct and hallucinated responses share the same prompt, history, and evidence, so a pre generation model should score that pair equally; its AUROC is 0.500, a useful check that it is not using the response.",
            "For a task that is decidable before generation, we formed a HaluEval counterfactual: the same public prompts are paired with either their own knowledge passage or another record's passage. PHC X reaches AUROC 0.705 on this context coverage task. This is evidence that context coverage transfers; it is not evidence that hallucination detection generalizes to independently generated responses."
        ]),
        ("VII. Limitations and Conclusion", [
            "The benchmark is synthetic and its labels are not independent human judgments. The leak controlled set is smaller than the standard benchmark, and category level results vary. External response level hallucination detection remains unvalidated. A blinded annotation packet and analysis pipeline are ready, but human annotation has not yet been performed. The model is decision support: a low score does not guarantee that an answer is factual.",
            "PHC X tests a practical hypothesis: conversation provenance, consistency, and evidence coverage can reveal risk before an assistant replies. The leak controlled results support that hypothesis on this synthetic task. Independent human evaluation is the next step needed to establish whether the approach is useful beyond the benchmark."
        ]),
    ]

    for heading, paragraphs in sections:
        _add_heading(doc, heading, level=1)
        for text in paragraphs:
            _add_body(doc, text)
        if heading == "IV. Evaluation":
            _add_table(doc,
                       ["Measure", "Standard RF", "Leak controlled XGBoost"],
                       [
                           ("Test prompts", "161", "107"),
                           ("AUROC", "0.987", "0.975"),
                           ("AUPRC", "0.992", "0.988"),
                           ("Prompt only AUROC", "0.852", "0.434"),
                           ("Template lookup AUROC", "0.808", "0.635"),
                           ("Action coverage", "52.8 percent", "50.5 percent"),
                       ],
                       "Table 1. Main and leak controlled benchmark results.")

    _add_heading(doc, "References", level=1)
    for i, ref in enumerate(_read_references(), 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.paragraph_format.space_after = Pt(1)
        _set_run(p.add_run(f"[{i}] {ref}"), size=8)

    # Apply black Times New Roman to every paragraph and table run, including
    # inherited styles, so there are no theme-colored headings or links.
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            _set_run(run, size=run.font.size.pt if run.font.size else 9.5,
                     bold=run.bold, italic=run.italic)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        _set_run(run, size=8, bold=run.bold, italic=run.italic)
    for section in doc.sections:
        for part in (section.header, section.footer):
            for paragraph in part.paragraphs:
                paragraph.text = ""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


def main() -> int:
    # Keep the historical command stable while using the template-faithful
    # implementation for current builds.
    from fill_conference_template import main as fill_template
    return fill_template()


if __name__ == "__main__":
    raise SystemExit(main())
