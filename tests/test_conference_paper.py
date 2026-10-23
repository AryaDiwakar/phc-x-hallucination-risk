"""Smoke checks for the separate IEEE A4 conference draft."""
import re

from scripts.build_conference_paper import ABSTRACT, PREAMBLE, _conference_tables, render


def test_conference_draft_uses_a4_ieee_two_column_class():
    source = render()
    assert r"\documentclass[conference,a4paper]{IEEEtran}" in source
    assert r"\begin{IEEEkeywords}" in source
    assert r"\input{conference_tables.tex}" in source
    assert r"\input{tables.tex}" not in source


def test_conference_title_and_abstract_avoid_math_symbols_and_notes():
    source = render()
    title = re.search(r"\\title\{(.*?)\}", source, re.S).group(1)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", source, re.S).group(1)
    for text in (title, abstract):
        assert "$" not in text
        assert r"\footnote" not in text
        assert r"\cite" not in text
        assert r"\%" not in text
    assert "Affiliation and contact details to be completed" in PREAMBLE


def test_conference_bibliography_is_numeric_style():
    source = render()
    assert not re.search(r"\\bibitem\[[^\]]+\]\{", source)
    assert r"\bibitem{selfcheckgpt}" in source


def test_conference_tables_are_sized_for_a4_columns():
    tables = _conference_tables()
    assert r"\begin{table*}" in tables
    assert r"\resizebox{\textwidth}{!}" in tables
    for label in ("tab:main", "tab:category", "tab:ablation", "tab:gates",
                  "tab:leakcontrol", "tab:external"):
        assert label in tables
    assert "XGBoost, depth 3" in tables
