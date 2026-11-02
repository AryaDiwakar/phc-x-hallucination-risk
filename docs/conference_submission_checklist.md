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

Outputs: `paper/conference_submission.docx` (filled from the supplied Word
template) and `paper/conference.pdf` (the IEEEtran LaTeX rendering). The Word
draft retains the template's title and three-author layout, A4 page geometry,
named paragraph styles, and two-column body. Results tables are sized to one
column. These are author-review drafts, not submission-ready yet.

## Template requirements handled

- A4 page size and two-column IEEE conference layout
- no template-instruction boilerplate copied into the paper
- title and abstract have no math, footnotes or citation commands
- keywords follow the abstract
- numeric IEEE-style citations (author-year labels removed from bibliography
  entries in this version)
- tables have captions above them and are constrained to one column; no
  table spans the full page
- figures use captions below; tables use captions above

## Author action required before submission

The author blocks use the supplied register numbers and the affiliation
`School of Computer Science and Engineering, VIT Vellore`. Author email addresses or ORCIDs were not supplied; add
them if the target conference requires author contact details.

Also verify the target conference's specific page limit, copyright notice,
required PDF eXpress check, and whether it accepts LaTeX output. The provided
Word template is guidance for layout; IEEEtran is a matching LaTeX rendering,
not a conversion of the `.docx` itself.

## Visual review still needed

The PDF compiles, but Tectonic reports underfull boxes in dense two-column
prose. Inspect the PDF at normal zoom for line breaks, float order and figure
legibility before submission. In particular, confirm the six tables and the
two-column figure placement meet the target conference's rules.
