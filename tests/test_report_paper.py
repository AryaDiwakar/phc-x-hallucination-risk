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
