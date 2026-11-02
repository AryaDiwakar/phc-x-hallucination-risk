"""Populate the supplied IEEE A4 Word template with the PHC-X paper draft.

The supplied file is Strict OOXML. We normalize a temporary copy only so
python-docx can edit it; page geometry and named styles come from that template.
The resulting manuscript is an AI-assisted editorial draft for author review,
not a claim of human authorship or completed human validation.
"""
from __future__ import annotations

import re
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "conference-template-a4.docx"
OUTPUT = ROOT / "paper" / "conference_submission.docx"
PAPER_TEX = ROOT / "paper" / "main.tex"

STRICT_NAMESPACES = {
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


def _transitional_copy(source: Path) -> Path:
    target = Path(tempfile.gettempdir()) / "phcx-ieee-template-transitional.docx"
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(
        target, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename.endswith((".xml", ".rels")):
                text = data.decode("utf-8", "replace")
                for strict, transitional in sorted(
                    STRICT_NAMESPACES.items(), key=lambda pair: -len(pair[0])
                ):
                    text = text.replace(strict, transitional)
                data = text.encode("utf-8")
            dst.writestr(info, data)
    return target


def _font(run, size: float | None = None, bold: bool | None = None,
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


def _black_tnr_styles(doc) -> None:
    for style in doc.styles:
        try:
            style.font.name = "Times New Roman"
            style.font.color.rgb = RGBColor(0, 0, 0)
            rpr = style._element.get_or_add_rPr()
            fonts = rpr.rFonts
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.insert(0, fonts)
            for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
                fonts.set(qn(f"w:{attr}"), "Times New Roman")
        except (AttributeError, TypeError):
            continue


def _wipe_body(doc) -> None:
    body = doc._element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def _p(doc, text: str, style: str = "Body Text", align=None,
       keep_next: bool = False):
    paragraph = doc.add_paragraph(style=style)
    if align is not None:
        paragraph.alignment = align
    paragraph.paragraph_format.keep_with_next = keep_next
    run = paragraph.add_run(text)
    _font(run)
    return paragraph


def _heading(doc, text: str, level: int = 1):
    style = {1: "Heading 1", 2: "Heading 2", 3: "Heading 3", 5: "Heading 5"}[level]
    align = WD_ALIGN_PARAGRAPH.CENTER if level in (1, 5) else WD_ALIGN_PARAGRAPH.LEFT
    paragraph = _p(doc, text.upper(), style=style, align=align, keep_next=True)
    for run in paragraph.runs:
        _font(run, bold=True, italic=run.italic)
    return paragraph


def _set_table_borders(table) -> None:
    pr = table._tbl.tblPr
    borders = pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single" if edge in ("top", "bottom", "insideH") else "nil")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), "000000")


def _cell(cell, text: str, header: bool = False, author: bool = False) -> None:
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if author or header else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(0)
    if author:
        p.style = "Author"
    elif header:
        p.style = "table head"
    else:
        p.style = "Normal"
    for line_no, line in enumerate(text.split("\n")):
        if line_no:
            p.add_run().add_break()
        run = p.add_run(line)
        _font(run, size=10 if author else 7.2, bold=header)


def _table(doc, number: str, title: str, headers: list[str], rows: list[tuple[str, ...]]) -> None:
    caption = _p(doc, f"TABLE {number}\n{title.upper()}", style="table head",
                 align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(3)
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    # A two-column IEEE body has a text column of about 3.38 inches. Keep every
    # table inside that column rather than allowing Word to span the page.
    width = Inches(3.34)
    col_width = Inches(3.34 / len(headers))
    for col in table.columns:
        col.width = col_width
    for cell in table.rows[0].cells:
        cell.width = col_width
    for cell, value in zip(table.rows[0].cells, headers):
        _cell(cell, value, header=True)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.width = col_width
            _cell(cell, value)
    tbl_pr = table._tbl.tblPr
    tbl_width = tbl_pr.find(qn("w:tblW"))
    if tbl_width is None:
        tbl_width = OxmlElement("w:tblW")
        tbl_pr.append(tbl_width)
    tbl_width.set(qn("w:w"), str(round(width.twips)))
    tbl_width.set(qn("w:type"), "dxa")
    _set_table_borders(table)
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(1)


def _references() -> list[str]:
    text = PAPER_TEX.read_text(encoding="utf-8")
    match = re.search(r"\\begin\{thebibliography\}.*?(.*?)\\end\{thebibliography\}", text, re.S)
    if not match:
        raise ValueError("Could not find the bibliography in paper/main.tex")
    entries = re.findall(
        r"\\bibitem(?:\[[^\]]*\])?\{[^}]+\}(.*?)(?=\\bibitem|$)",
        match.group(1), re.S,
    )
    result = []
    for entry in entries:
        entry = entry.replace("\n", " ")
        entry = re.sub(r"\\url\{([^}]*)\}", r"\1", entry)
        entry = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", "", entry)
        entry = entry.replace("{", "").replace("}", "").replace("~", " ")
        result.append(re.sub(r"\s+", " ", entry).strip())
    return result


def build() -> Path:
    doc = Document(_transitional_copy(TEMPLATE))
    _wipe_body(doc)
    _black_tnr_styles(doc)
    doc.styles["paper title"].font.size = Pt(20)

    # Keep the supplied A4 geometry. The first section is the template's
    # full-width title/author area; the body uses its IEEE two-column settings.
    front = doc.sections[0]
    front.page_width, front.page_height = Inches(8.27), Inches(11.69)
    front.left_margin = front.right_margin = Inches(0.62)
    front.top_margin, front.bottom_margin = Inches(0.375), Inches(1.0)
    front.header_distance, front.footer_distance = Inches(0.25), Inches(0.5)
    front_cols = front._sectPr.find(qn("w:cols"))
    if front_cols is not None:
        front_cols.set(qn("w:num"), "1")

    title = _p(
        doc,
        "PHC-X: Provenance-Aware Pre-Generation Hallucination Risk Prediction for Multi-Turn Conversations".upper(),
        style="paper title", align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=True,
    )
    title.paragraph_format.keep_together = True
    for run in title.runs:
        _font(run, size=20, bold=True)

    author_table = doc.add_table(rows=1, cols=3)
    author_table.autofit = False
    author_width = Inches(7.03 / 3)
    names = [
        "Arya D\n23BDS0023\nSchool of Computer Science\nand Engineering\nVIT Vellore",
        "Jeswanth G\n23BDS0212\nSchool of Computer Science\nand Engineering\nVIT Vellore",
        "N Sai Baba\n23BDS0019\nSchool of Computer Science\nand Engineering\nVIT Vellore",
    ]
    for col, value in zip(author_table.columns, names):
        col.width = author_width
    for cell, value in zip(author_table.rows[0].cells, names):
        cell.width = author_width
        _cell(cell, value, author=True)
    # Author-grid tables are intentionally borderless, as in the template.
    _set_table_borders(author_table)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = author_table._tbl.tblPr.find(qn("w:tblBorders")).find(qn(f"w:{edge}"))
        el.set(qn("w:val"), "nil")

    body = doc.add_section(WD_SECTION.CONTINUOUS)
    body.page_width, body.page_height = Inches(8.27), Inches(11.69)
    body.left_margin = body.right_margin = Inches(0.63)
    body.top_margin, body.bottom_margin = Inches(0.75), Inches(1.0)
    body.header_distance, body.footer_distance = Inches(0.5), Inches(0.5)
    cols = body._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        body._sectPr.append(cols)
    cols.set(qn("w:num"), "2")
    cols.set(qn("w:space"), "18pt")

    abstract = doc.add_paragraph(style="Abstract")
    abstract.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    _font(abstract.add_run("Abstract—"), bold=True)
    _font(abstract.add_run(
        "Pre-generation screening asks whether a conversational assistant is at risk of producing an unsupported next reply before that reply is written. We present PHC-X, a black-box framework that combines conversation history, claim provenance, cross-turn consistency checks, and evidence coverage to estimate this risk and select an intervention. On the standard synthetic PHC-Conversation benchmark (1,100 prompts in 238 near-duplicate families), a calibrated random forest reaches a test AUROC of 0.987; however, a prompt-template lookup baseline reaches 0.808, exposing a real benchmark shortcut. We therefore report a separate leak-controlled benchmark in which all categories draw from a shared prompt pool. On its 107-example held-out test split, the training-only, family-grouped model-selection procedure selects depth-four XGBoost, which reaches AUROC 0.975 and AUPRC 0.988. The prompt-only ablation falls to 0.434, while history, provenance, contradiction, and evidence features restore performance. External tests do not support broad response-level generalization: DiaHalu performance is poor, and matched HaluEval responses are indistinguishable before generation. A context-swapped HaluEval task reaches AUROC 0.705, supporting transfer only for context-coverage assessment. The benchmark is synthetic, human annotation is pending, and these results are not a claim of state-of-the-art performance."
    ))
    keywords = doc.add_paragraph(style="Keywords")
    _font(keywords.add_run("Index Terms—"), bold=True, italic=True)
    _font(keywords.add_run(
        "hallucination risk, conversational systems, provenance, evidence coverage, calibration, intervention"
    ), italic=True)

    _heading(doc, "I. INTRODUCTION")
    _p(doc,
       "Conversational assistants are often evaluated after an answer has been produced: a second model checks the response, several samples are compared, or a verifier searches for supporting evidence. SelfCheckGPT and semantic-uncertainty methods use disagreement among samples [1]–[3], while chain-of-verification and retrieval-and-revision systems inspect an existing draft [5], [6]. These approaches can detect some errors, but they act after the user has already received the answer. A pre-generation decision has a different goal: assess the pending turn from the information available in the dialogue and choose whether to answer, retrieve, ask a clarifying question, or abstain.")
    _p(doc,
       "Most pre-decoding detectors use a model's internal activations or attention and assess an isolated prompt [7]–[13]. That design is not always available when the language model is a remote service, and it does not directly represent how risk accumulates across turns. An assistant can repeat an earlier unsupported statement as if it were evidence, silently switch the entity under discussion, rely on stale information, or accept a false premise embedded in a follow-up. The relevant unit is therefore not only the current sentence; it is the current sentence in its conversational context.")
    _p(doc,
       "PHC-X treats the conversation as an auditable state. Before generation, it records who introduced each claim and whether the claim is supported, compares the pending prompt with prior turns and retrieved passages, and maps the resulting signals to a calibrated risk score and policy action. The system is deliberately black-box at serving time: it does not require access to hidden states, logits, or a particular model provider. Its contribution is a framework and a carefully qualified benchmark study, not a claim that an AUROC on synthetic examples establishes reliable real-world hallucination prevention.")
    _p(doc,
       "This paper makes four contributions. First, it describes a provenance-aware, conversation-level feature pipeline with five deterministic consistency checks and evidence-coverage scoring. Second, it presents a calibrated risk-to-action policy with explicit answerability constraints. Third, it reports both the original synthetic benchmark and a separately constructed leak-controlled variant, including the prompt-only and template-lookup probes. Fourth, it reports negative and positive external tests separately: response-level results do not transfer, while a context-coverage counterfactual does. Human annotation remains a necessary next step.")

    _heading(doc, "II. RELATED WORK")
    _p(doc,
       "Post-generation factuality methods include sampling-based consistency checks, semantic entropy, and verification or revision pipelines [1]–[6]. They can inspect the answer itself, which gives them information that a genuinely pre-generation detector cannot use. Results from these approaches are consequently not directly interchangeable with next-turn risk prediction.")
    _p(doc,
       "Pre-generation work has explored hidden-state classifiers, entity representations, attention probes, and confidence-based routing [7]–[13]. These methods motivate early intervention, but generally make use of internal model signals or score a prompt independently. PHC-X instead focuses on conversation state and evidence available before the next response. We do not claim that this makes the method universally stronger; it makes the problem definition different.")
    _p(doc,
       "Dialogue benchmarks such as DiaHalu and HalluHard study errors in multi-turn interactions [14], [15]. Other work accumulates risk over a session, limits memory writes, or evaluates stale-memory conflicts [16]–[19]. Studies of false presuppositions also show that systems may accept unsupported assumptions in a question [20]. Long-form factuality probes and attention-sink signals extend the broader study of internal model cues [21], [22]. These lines of work provide relevant tasks and mechanisms, but do not by themselves establish that a conversation-only score can forecast a particular model's next answer. We use them to frame the gap and to design external checks, not as evidence that PHC-X has solved it.")

    _heading(doc, "III. PHC-X FRAMEWORK")
    _heading(doc, "A. Processing a pending turn", level=2)
    _p(doc,
       "At each request, PHC-X receives a conversation history, the pending user prompt, and an optional local evidence corpus. It constructs claim-level memory from the turns, computes deterministic consistency and provenance features, retrieves candidate passages using a hashed n-gram term-frequency index, and assembles a 54-dimensional feature vector. A fitted model estimates risk; a validation-calibrated policy converts the score into an action. The sequence ends before an assistant response is generated, so response quality itself is not an input to this decision.")
    _heading(doc, "B. Provenance-aware memory", level=2)
    _p(doc,
       "Each conversation turn is retained with its speaker and source. Candidate claims are tagged as user-provided, assistant-generated, evidence-derived, or model-derived, with a trust score and verification state. This distinction matters because repetition is not corroboration: an assistant statement that was unsupported when first produced should not become trusted merely because the user later asks a question that repeats it. The contamination feature measures how strongly the pending prompt depends on such an unverified assistant claim.")
    _p(doc,
       "For example, consider a conversation in which the assistant previously stated that dataset Atlas A contains a particular number of samples. If a source document verifies the statement, a follow-up asking for the count has a different evidential status than the same follow-up after an unsourced assistant assertion. PHC-X represents that difference in the memory provenance rather than inferring reliability from fluent wording. The example illustrates the feature design; it is not a separate human evaluation result.")
    _heading(doc, "C. Consistency and evidence features", level=2)
    _p(doc,
       "Five rule-based engines produce bounded signals between zero and one. Contradiction checks compare a pending assertion with relevant prior statements while avoiding treating genuine questions as factual claims. Entity-drift checks look for changes in typed slots such as dataset, model, or organization. Temporal checks compare dates and freshness requirements with stored claims. Presupposition checks identify questions that assume an event, change, or quantity not established in the history. Referential-ambiguity checks identify references with multiple plausible antecedents and avoid flagging references that resolve cleanly.")
    _p(doc,
       "The local retriever ranks evidence passages with hashed n-gram term-frequency cosine similarity. The evidence component considers coverage, contradiction, staleness, and whether apparent support comes only from model-generated claims. It is deterministic and can run offline, but it is not a semantic search system: paraphrases and lexical mismatch can reduce recall. Evidence coverage above the configured floor dampens the risk score, supporting a non-blocking generate-with-evidence action when the retrieved material addresses the question.")
    _heading(doc, "D. Risk estimation, calibration, and policy", level=2)
    _p(doc,
       "The feature vector groups prompt, history, provenance, consistency, evidence, and optional model-specific signals. The deployed default is black-box; the white-box group is not available in the standard offline serving path. The standard benchmark uses the prespecified random forest. For the leak-controlled experiment, candidate learners were compared by five-fold GroupKFold on training families only, across seeds 42, 123, and 2024. XGBoost with 300 trees, maximum depth four, and learning rate 0.03 was selected by mean raw AUROC. Validation was not used for candidate ranking and the held-out test split was not touched during selection.")
    _p(doc,
       "Platt scaling is fitted on validation predictions, and action thresholds are searched on validation data under a cost-sensitive objective and a minimum-coverage constraint. Calibration and threshold selection are therefore distinct from the training-only learner sweep. The policy offers generate, generate with evidence, retrieve, clarify, and abstain (the default interface may combine the last two into clarify-or-abstain). Blocking actions require an explicit user override in the browser extension; the extension fails open if the local service is unavailable. These deployment choices reduce the risk of silent blocking, but do not make a model score a truth guarantee.")

    _heading(doc, "IV. BENCHMARKS AND EXPERIMENTAL PROTOCOL")
    _heading(doc, "A. Standard PHC-Conversation", level=2)
    _p(doc,
       "The standard PHC-Conversation benchmark contains 1,100 synthetic prompts grouped into 238 near-duplicate families, split into 768 training, 171 validation, and 161 test rows. The eleven categories comprise ten risk types and a short-benign control. Variants from the same family remain in one partition. A construction-derived label and a rule-verified label are combined by a union for the main task; their row-level agreement is 75.18 percent. Duplicate, semantic-duplicate, family-overlap, evidence-overlap, and variant-crossing checks pass. The synthetic generation process makes the task reproducible, but also creates a risk that a classifier learns the generator's wording rather than dialogue evidence.")
    _heading(doc, "B. Leak-controlled benchmark", level=2)
    _p(doc,
       "We measured this shortcut directly. In the standard test split, 84.2 percent of the 57 distinct templates are uniquely associated with a category. A lookup rule that reads only the prompt template, not the conversation, reaches AUROC 0.808. That result is too large to ignore when interpreting the standard model's AUROC of 0.987.")
    _p(doc,
       "The leak-controlled construction uses one shared prompt pool across seven context-borne categories: contradictory context, entity ambiguity, evidence conflict, long-context degradation, model-memory contamination, supported context, and unsupported context. It contains 700 rows in 230 families. The four categories whose label signal is intrinsically expressed in the prompt—presupposition, referential ambiguity, freshness cue, and benign first turn—are excluded rather than relabeled as context risks. The shared pool has three templates and none uniquely identifies a category. The template-lookup AUROC falls to 0.635. Gold and verifier labels agree on 64.57 percent of this variant, so this benchmark also retains meaningful label uncertainty.")
    _p(doc,
       "For model selection, the training split alone is divided into five family-grouped folds for each of three seeds, yielding fifteen fold-seed evaluations per candidate. The test split contains 107 rows (72 positive and 35 negative); validation is used only for calibration and policy thresholds. The training-only sweep selected depth-four XGBoost with mean fold AUROC 0.9796 (standard deviation 0.0058) and AUPRC 0.9885 (0.0045). The existing random forest scored AUROC 0.9563 (0.0172) on the same grouped selection procedure. We report this change explicitly: the standard benchmark and leak-controlled benchmark use different selected classifiers.")

    _heading(doc, "C. Measures and comparisons", level=2)
    _p(doc,
       "We report AUROC and AUPRC for ranking, F1/precision/recall at the validation-selected threshold, Brier score and ten-bin expected calibration error (ECE) for probability quality, and action-level coverage and positive-case capture for the policy. The standard test metrics include 1,000 bootstrap intervals for the model and the heuristic baseline. Correlated DeLong testing compares AUROCs; exact McNemar testing compares thresholded decisions. The DeLong comparison with the heuristic is not statistically significant in this test sample (p = 0.452), so the point estimates are descriptive and should not be presented as a proven significant AUROC improvement.")

    _heading(doc, "V. RESULTS")
    _heading(doc, "A. Standard benchmark: strong score, exposed shortcut", level=2)
    _p(doc,
        "On the 161-row standard test split, the calibrated random forest achieves AUROC 0.9868, AUPRC 0.9925, and F1 0.947 at its validation-selected threshold of 0.31. Precision is 0.925 and recall is 0.971. The 95 percent bootstrap intervals are 0.972–0.996 for AUROC and 0.983–0.998 for AUPRC. Calibration improves ECE from 0.0917 before scaling to 0.0503 after scaling, and Brier score from 0.0595 to 0.0489. Across three seeds, mean AUROC is 0.98674 (SD 0.00021) and mean AUPRC is 0.99253 (SD 0.00014). The heuristic baseline reaches AUROC 0.724, while logistic regression on the same features reaches 0.960.")
    _table(doc, "I", "Selected test-set ranking metrics", ["Evaluation", "AUROC", "AUPRC", "F1"], [
        ("Standard RF, n=161", "0.987", "0.992", "0.947"),
        ("Standard template lookup", "0.808", "0.858", "0.837"),
        ("Standard heuristic", "0.724", "0.848", "0.711"),
        ("Leak-controlled XGBoost, n=107", "0.975", "0.988", "0.916"),
        ("Leak-controlled template lookup", "0.635", "0.779", "0.804"),
    ])
    _p(doc,
       "Those figures describe performance on the constructed split, not performance on naturally occurring hallucinations. The prompt-only ablation reaches AUROC 0.852, and the template lookup reaches 0.808 without reading history. The unusually high standard score is therefore partly entangled with prompt design. The family split and leakage checks address duplicate contamination, but they do not remove a category-specific lexical cue present across distinct families. The standard score remains useful as a benchmark result only when the shortcut is displayed alongside it.")
    _table(doc, "II", "Standard benchmark calibration and operating point", ["Metric", "Calibrated RF", "Uncalibrated RF"], [
        ("AUROC / AUPRC", "0.987 / 0.992", "0.987 / 0.992"),
        ("F1; precision; recall", "0.947; 0.925; 0.971", "0.935; 0.886; 0.990"),
        ("ECE; Brier", "0.050; 0.049", "0.092; 0.060"),
        ("Threshold", "0.31", "0.31"),
    ])

    _heading(doc, "B. Leak-controlled benchmark: context signal", level=2)
    _p(doc,
       "On the held-out leak-controlled test split, the selected XGBoost model obtains AUROC 0.9750, AUPRC 0.9882, and F1 0.916 at threshold 0.09. It identifies 71 of 72 positive examples (recall 0.986), while correctly classifying 23 of 35 negatives (specificity 0.657). Test AUROC across three retraining seeds is 0.9753 with standard deviation 0.0010. The grouped training-fold selection result and the test result answer different questions: the former compares candidates without seeing test data; the latter is the final estimate for the selected candidate on the reserved split.")
    _p(doc,
       "The ablation ladder helps locate the benchmark signal. Prompt-only features reach AUROC 0.434. Adding history raises the result to 0.749; provenance raises it to 0.779; adding contradiction checks gives 0.783; and adding evidence coverage brings it to 0.975. The full black-box configuration is also 0.975. This pattern is consistent with the task definition: because labels are context-borne, the dialogue and its evidence matter more than the shared prompt wording. It is not proof that the same signal will appear in uncurated conversations, where topic, source quality, and user intent are less controlled.")
    _table(doc, "III", "Training-only, family-grouped model sweep", ["Candidate", "AUROC mean ± SD", "AUPRC mean ± SD"], [
        ("XGBoost d4; 300 trees, lr .03 (selected)", "0.9796 ± 0.0058", "0.9885 ± 0.0045"),
        ("XGBoost d3; 300 trees, lr .03", "0.9779 ± 0.0068", "0.9874 ± 0.0053"),
        ("LightGBM d4; 300 trees, lr .03", "0.9773 ± 0.0067", "0.9872 ± 0.0056"),
        ("XGBoost d3; 500 trees, lr .01", "0.9753 ± 0.0080", "0.9861 ± 0.0059"),
        ("LightGBM d3; 300 trees, lr .03", "0.9724 ± 0.0082", "0.9845 ± 0.0066"),
        ("XGBoost d2; 300 trees, lr .03", "0.9688 ± 0.0130", "0.9823 ± 0.0091"),
        ("LightGBM d2; 300 trees, lr .03", "0.9662 ± 0.0136", "0.9804 ± 0.0100"),
        ("RF unlimited depth; 500 trees, leaf 2", "0.9659 ± 0.0132", "0.9805 ± 0.0099"),
        ("RF depth 8; 500 trees, leaf 2", "0.9641 ± 0.0139", "0.9792 ± 0.0107"),
        ("Existing RF; 300 trees, depth 8, leaf 5", "0.9563 ± 0.0172", "0.9744 ± 0.0138"),
    ])
    _table(doc, "IV", "Leak-controlled feature ablation", ["Feature set", "AUROC", "AUPRC"], [
        ("Prompt only", "0.434", "—"),
        ("Prompt + history", "0.749", "—"),
        ("History + provenance", "0.779", "—"),
        ("Plus contradiction", "0.783", "—"),
        ("Plus evidence", "0.975", "—"),
        ("Full black-box", "0.975", "0.988"),
        ("Full plus white-box signals", "0.975", "—"),
    ])

    _heading(doc, "C. Action policy and error profile", level=2)
    _p(doc,
       "At the standard benchmark's validation-tuned operating point, the policy leaves 52.8 percent of test turns answerable and catches 97.1 percent of positive cases. The positive rate among turns assigned the direct GENERATE action is 2.08 percent (one positive among 48 such turns), compared with a 63.35 percent test base rate. The leak-controlled policy leaves 50.5 percent answerable and catches 98.6 percent of positives; no positive occurs in its direct-GENERATE group on this 107-row test sample. These are finite-sample operating-point results, not guarantees of zero risk in deployment. A coverage constraint matters because an always-block policy could trivially catch every positive while being unusable.")
    _table(doc, "V", "Policy operating points on held-out tests", ["Measure", "Standard RF", "Leak-controlled XGB"], [
        ("Answerable coverage", "52.8%", "50.5%"),
        ("Positive cases caught", "97.1%", "98.6%"),
        ("Blocked share", "47.2%", "49.5%"),
        ("Direct-GENERATE risk", "2.08% (1/48)", "0% (0 observed)"),
        ("Test positive base rate", "63.35%", "67.29%"),
    ])
    _p(doc,
       "Category-level results should be read cautiously because several categories have only ten to twenty test examples and some contain a single label class, for which AUROC is undefined. Leave-one-category-out stress testing has a mean AUROC of 0.804 across nine categories with a standard deviation of 0.285; it ranges from 0.195 on supported context to 1.000 on several categories. This spread is a warning against summarizing transfer as uniformly strong. Cross-label evaluation is also asymmetric: training on construction labels and scoring against verifier labels gives AUROC 0.779, while the reverse direction gives 0.671. The underlying label sources agree on only 75.18 percent of standard-benchmark rows.")

    _heading(doc, "VI. EXTERNAL EVALUATION")
    _p(doc,
       "External data do not provide one universal test of this framework because the target must be decidable before the response exists. We therefore report three deliberately distinct checks and do not pool them into a single generalization score.")
    _p(doc,
       "DiaHalu is aligned to the user turn that elicited the annotated response, with preceding turns used as history and later turns excluded. Across 5,923 examples, the synthetic-trained model has raw AUROC 0.431 (95 percent bootstrap interval 0.404–0.458). The labels are imbalanced, and error timing is associated with dialogue position; after stratifying by history length, pooled AUROC is 0.510. At the chosen threshold, the policy blocks 79.3 percent of examples, leaving only 20.7 percent coverage, while the non-blocked hallucination rate is 10.99 percent versus an 8.26 percent base rate. This is not evidence of useful transfer; it is a negative result that exposes poor ranking and an unsatisfactory operating point.")
    _p(doc,
       "The HaluEval dialogue purity check pairs supported and hallucinated responses that share the same prompt, history, and evidence. A system that runs before generation should not be able to separate that pair from those inputs alone. PHC-X obtains AUROC 0.500 on 20,000 rows (95 percent interval 0.493–0.507), matching that expectation. This result is a check against response leakage, not a successful hallucination-detection result.")
    _p(doc,
       "A separate HaluEval counterfactual changes the available knowledge passage while keeping the public prompt: one condition uses its own passage and the other uses a passage from another record. This makes context support the target rather than the truthfulness of an already-written response. On 20,000 balanced examples, AUROC is 0.705 (95 percent interval 0.698–0.713) and AUPRC is 0.659. The result provides limited evidence that the context-coverage feature transfers to this construction. It does not establish generalization to independently generated assistant responses, and the label construction remains a controlled counterfactual rather than a human assessment of answer factuality.")
    _table(doc, "VI", "External evaluations without retraining or recalibration", ["Task", "n", "AUROC", "AUPRC"], [
        ("DiaHalu, turn-aligned", "5,923", "0.431", "0.072"),
        ("DiaHalu, history-stratified", "5,923", "0.510", "—"),
        ("HaluEval matched responses", "20,000", "0.500", "0.500"),
        ("HaluEval context counterfactual", "20,000", "0.705", "0.659"),
    ])

    _heading(doc, "VII. IMPLEMENTATION AND RESPONSIBLE USE")
    _p(doc,
        "The research system includes a command-line interface for dataset generation, training, evaluation, calibration, ablation, and reporting; a local REST API for analysis, batch scoring, policy queries, and decision logging; and a Manifest V3 browser extension with a visible override for blocking actions. A dashboard exposes scores, policy thresholds, and benchmark metrics. The pipeline is deterministic and seeded, and does not require a hosted inference call for risk scoring. A warm local pipeline measured over 25 calls has median latency 19.58 ms and 95th-percentile latency 27.89 ms, including feature attribution; these machine-local timings are not a cross-device service-level guarantee. The present paper evaluates the feature/model pipeline, not end-user outcomes or a production deployment study.")
    _p(doc,
       "The interface should present a risk estimate as a prompt to gather evidence or clarify intent, not as a verdict that the user or assistant is wrong. False positives can interrupt useful work; false negatives can permit unsupported answers. Logging should be minimized and handled under the deploying institution's privacy requirements. Because blocking is potentially costly, the extension makes blocking explicit and overridable and fails open on service failure. These safeguards are design choices; their effects on user trust and answer quality have not been measured here.")

    _heading(doc, "VIII. LIMITATIONS")
    _p(doc,
       "The central limitation is the benchmark. PHC-Conversation is synthetic, generated from a controlled set of families, and labeled by construction plus an automated verifier rather than independent human annotators. Agreement is 75.18 percent in the standard dataset and 64.57 percent in the leak-controlled variant. The latter has only 107 test rows, so even a high ranking score has substantial uncertainty about performance on a broader population. Repeated seeds measure training variability, not independent data variability.")
    _p(doc,
       "The external tests reinforce rather than remove this limitation. DiaHalu is a negative transfer result; matched HaluEval responses are intentionally indistinguishable; and the positive HaluEval counterfactual tests context coverage, not response hallucination detection. The cross-category results are heterogeneous, and the standard benchmark contains a measurable prompt shortcut. The hashed lexical retriever can miss paraphrases, and the policy's coverage/recall trade-off may shift under a different class balance or user population. We have not measured whether interventions reduce factual errors, improve task completion, or impose acceptable latency in live use.")
    _p(doc,
       "A blinded human annotation packet and analysis pipeline have been prepared, but no human annotations have been collected. A simulator-based dry run is not human validation and is not reported as such. Before any deployment claim, the next study should use blinded, stratified annotators, report agreement and adjudication, include independent real conversations, evaluate answer quality after intervention, and test coverage and error costs under a prespecified protocol. The authors should also verify dataset permissions and their institution's privacy and ethics requirements.")

    _heading(doc, "IX. CONCLUSION")
    _p(doc,
       "PHC-X tests whether a system can estimate next-turn risk from conversation provenance, consistency, and evidence before generating a reply. The standard synthetic benchmark gives a strong score but also reveals a prompt-template shortcut; the leak-controlled benchmark is therefore the more relevant result for conversation-based discrimination. There, family-grouped, training-only model selection chooses depth-four XGBoost, and the held-out result is AUROC 0.975 on 107 examples. Ablations show that the shared prompt alone does not explain that score, while evidence coverage is associated with the largest increase in the constructed task.")
    _p(doc,
       "The conclusion is deliberately bounded. Current evidence supports a reproducible prototype and a context-coverage signal on a controlled counterfactual. It does not establish reliable real-world hallucination prevention, broad external generalization, or human-validated performance. Independent human evaluation and intervention-outcome studies are required before those claims can be made.")

    _heading(doc, "REFERENCES", level=5)
    for index, reference in enumerate(_references(), 1):
        paragraph = doc.add_paragraph(style="references")
        paragraph.paragraph_format.left_indent = Inches(0.18)
        paragraph.paragraph_format.first_line_indent = Inches(-0.18)
        run = paragraph.add_run(f"[{index}] {reference}")
        _font(run, size=8)

    # Remove template header/footer examples and make every visible run black
    # Times New Roman without overriding template paragraph sizes/spacing.
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            _font(run, size=run.font.size.pt if run.font.size else None,
                 bold=run.bold, italic=run.italic)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        _font(run, size=run.font.size.pt if run.font.size else None,
                             bold=run.bold, italic=run.italic)
    for section in doc.sections:
        for part in (section.header, section.footer):
            for paragraph in part.paragraphs:
                paragraph.text = ""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    return OUTPUT


def main() -> int:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")
    path = build()
    print(f"Wrote filled template draft: {path}")
    print("A4 template styles; two-column body; black Times New Roman; tables fit one column.")
    print("Author affiliations and register numbers are filled; add emails/ORCIDs if required by the venue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
