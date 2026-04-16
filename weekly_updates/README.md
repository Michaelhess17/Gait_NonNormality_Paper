# Weekly Progress Updates

Create one Quarto file per week in this folder.

Recommended naming pattern:

- `YYYY-MM-DD_progress.qmd`

Start from the provided template:

- `TEMPLATE.qmd`

On each push, GitHub Actions renders every `*.qmd` file in this folder to HTML and PDF and uploads those outputs as workflow artifacts.

If GitHub Pages is enabled for the repository, rendered outputs are also published at:

- Manuscript HTML: `/`
- Manuscript PDF: `/index.pdf`
- Weekly updates index: `/weekly_updates/`
- Individual weekly files: `/weekly_updates/<name>.html` and `/weekly_updates/<name>.pdf`
