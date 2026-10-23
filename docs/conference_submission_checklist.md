# Conference-format draft checklist

The supplied `conference-template-a4.docx` is an IEEE-style A4 Word template.
The repository's existing `paper/main.tex` remains the research draft; the
separate `paper/conference.tex` is generated in IEEEtran's A4 conference,
two-column layout so the existing draft is preserved.

## Build

```bash
python3 scripts/build_conference_paper.py
cd paper
tectonic -X compile conference.tex
```

The generated `conference.pdf` is a layout draft, not submission-ready yet.

## Template requirements handled

- A4 page size and two-column IEEE conference layout
- no template-instruction boilerplate copied into the paper
- title and abstract have no math, footnotes or citation commands
- keywords follow the abstract
- numeric IEEE-style citations (author-year labels removed from bibliography
  entries in this version)
- tables are regenerated at conference width; wide tables use a two-column
  float and are scaled to the text width
- figures use captions below; tables use captions above

## Author action required before submission

The source intentionally leaves `Affiliation and contact details to be
completed`. The repository does not identify the authors' official
departments, institution name, city, or submission email/ORCID. Replace that
placeholder using the actual author information; do not submit it as-is.

Also verify the target conference's specific page limit, copyright notice,
required PDF eXpress check, and whether it accepts LaTeX output. The provided
Word template is guidance for layout; IEEEtran is a matching LaTeX rendering,
not a conversion of the `.docx` itself.

## Visual review still needed

The PDF compiles, but Tectonic reports underfull boxes in dense two-column
prose. Inspect the PDF at normal zoom for line breaks, float order and figure
legibility before submission. In particular, confirm the six tables and the
two-column figure placement meet the target conference's rules.
