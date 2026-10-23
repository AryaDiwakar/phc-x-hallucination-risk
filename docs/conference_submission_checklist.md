# Conference-format draft checklist

The supplied `conference-template-a4.docx` is an IEEE-style A4 Word template.
The repository's existing `paper/main.tex` remains the research draft. The
separate Word draft is filled using the supplied template's A4 dimensions,
Times New Roman, black text, and two-column body layout; the LaTeX twin uses
IEEEtran. Both preserve the research draft.

## Build

```bash
python3 scripts/build_conference_docx.py
python3 scripts/build_conference_paper.py
cd paper
tectonic -X compile conference.tex
```

Outputs: `paper/conference_submission.docx` (the filled Word template) and
`paper/conference.pdf` (its IEEEtran LaTeX twin). They are drafts, not
submission-ready yet.

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

The draft intentionally leaves `Affiliation and contact details to be
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
