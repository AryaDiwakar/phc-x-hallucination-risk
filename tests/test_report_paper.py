"""Report + paper pipeline tests (static rendering, no LaTeX engine needed)."""
import json
import re
from pathlib import Path

import pytest

from phcx.experiments.paper import render_macros, render_tables, run_paper

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def stats():
    path = ROOT / "reports" / "stats.json"
    if not path.exists():
        pytest.skip("reports/stats.json not generated yet (run `phc report`)")
    return json.loads(path.read_text())


def test_paper_references_only_defined_macros(stats):
    """Every ``\\macro`` used in main.tex must be defined in macros.tex.

    A renamed macro leaves a dangling reference that only shows up when
    LaTeX compiles, which is far too late to notice.
    """
    main = ROOT / "paper" / "main.tex"
    macros = ROOT / "paper" / "macros.tex"
    if not main.exists() or not macros.exists():
        pytest.skip("paper sources not generated yet (run `phc paper`)")

    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macros.read_text()))
    body = main.read_text()
    # drop the definitions and includes, then collect the control words used
    body = re.sub(r"\\newcommand\{[^}]*\}\{[^}]*\}", "", body)
    body = re.sub(r"\\(input|include|usepackage|documentclass|label|ref|cite|begin|end)\b", "", body)
    used = set(re.findall(r"\\([A-Za-z]+)", body))

    # control words that come from the document class or standard packages
    known = {
        "textbf", "textit", "emph", "texttt", "textsc", "textbf", "small",
        "centering", "caption", "label", "ref", "toprule", "midrule",
        "bottomrule", "hline", "left", "right", "le", "le", "leq", "geq",
        "times", "pm", "in", "frac", "approx", "pm", "pm", "cdots", "ldots",
        "dots", "rightarrow", "to", "and", "or", "not", "item", "section",
        "subsection", "text", "mathrm", "mathbf", "mathbb", "mathcal",
        "newcommand", "renewcommand", "newcommand", "documentclass", "usepackage",
        "geometry", "amsmath", "amssymb", "mathptmx", "caption", "hyperref",
        "booktabs", "newcommand", "def", "frac", "quad", "qquad", "hskip",
        "vspace", "hspace", "newline", "linebreak", "centering", "centering",
        "bf", "it", "rm", "sf", "tt", "sc", "sl", "em", "LaTeX", "TeX",
        "alpha", "beta", "gamma", "tau", "sigma", "mu", "lambda", "phi",
        "epsilon", "delta", "theta", "rho", "eta", "zeta", "kappa", "omega",
        "infty", "partial", "nabla", "sum", "prod", "int", "langle",
        "rangle", "lfloor", "rfloor", "lceil", "rceil", "setminus",
        "subseteq", "supseteq", "in", "notin", "cup", "cap", "oplus",
        "otimes", "sim", "simeq", "approx", "equiv", "propto", "mid",
        "max", "min", "arg", "exp", "log", "sin", "cos", "tan", "lim",
        "sup", "inf", "det", "dim", "ker", "deg", "gcd", "bmod", "pmod",
        # document structure and packages used by the paper
        "author", "title", "date", "maketitle", "bibitem", "includegraphics",
        "linewidth", "geq", "leq", "cdot", "ss", "ge", "hspace", "vspace",
    }
    undefined = sorted(used - defined - known)
    assert not undefined, f"main.tex uses undefined macros: {undefined}"


def test_macros_are_valid_latex_command_names(stats):
    tex = render_macros(stats)
    # Match *every* definition, not just the well-formed ones: TeX reads a
    # control word as letters only, so a name containing a digit or an
    # underscore is silently split and redefines the part before it.
    all_names = re.findall(r"\\newcommand\{\\([^}]*)\}", tex)
    assert all_names, "no macros rendered"
    bad = [n for n in all_names if not re.fullmatch(r"[A-Za-z]+", n)]
    assert not bad, f"TeX control words must be letters only, got: {bad}"
    names = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex)
    for required in ("auroc", "msmean", "lathigh", "gatestatus", "pdelong"):
        assert required in names


def test_macros_contain_headline_numbers(stats):
    tex = render_macros(stats)
    assert f"{stats['model']['auroc']:.3f}" in tex
    assert "PASS" in tex or "FAIL" in tex


def test_tables_escape_underscores_and_render(stats):
    tex = render_tables(stats)
    assert "\\begin{tabular}" in tex
    assert "tab:main" in tex
    # category / action names contain underscores that would break TeX
    assert "_" not in tex.replace("\\_", ""), "unescaped underscore in table"


def test_main_tex_cites_only_defined_bibitems():
    main = (ROOT / "paper" / "main.tex").read_text()
    cited = {k.strip() for c in re.findall(r"\\cite\{([^}]*)\}", main)
             for k in c.split(",")}
    bib = set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]*)\}", main))
    bib |= set(re.findall(r"\\bibitem\{([^}]*)\}", main))
    assert cited, "paper cites nothing"
    assert cited <= bib, f"undefined citations: {sorted(cited - bib)}"


def test_run_paper_without_compile(tmp_path, stats):
    out = run_paper(stats_path=ROOT / "reports" / "stats.json",
                    paper_dir=tmp_path, figures_dir=ROOT / "reports" / "figures",
                    compile=False, show_progress=False)
    assert (tmp_path / "macros.tex").exists()
    assert (tmp_path / "tables.tex").exists()
    assert out["compiled"] is False
    assert out["figures"], "no figures copied"


def test_run_paper_requires_stats(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_paper(stats_path=tmp_path / "missing.json", paper_dir=tmp_path,
                  compile=False, show_progress=False)
