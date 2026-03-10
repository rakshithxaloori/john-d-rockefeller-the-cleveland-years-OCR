# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|
| 2026-03-10 | self | Assumed `pip install --user` would work with the system Python | Use a repo-local virtualenv in this workspace; Homebrew Python is PEP 668 managed |
| 2026-03-10 | self | Used a bad `sed` expression while listing repo files | Keep file discovery commands simple here; `find` and `rg --files` are enough |
| 2026-03-10 | self | Expected the full 167-screenshot OCR pass to finish quickly enough for an in-turn verification run | Treat the full dataset pass as a separate long-running validation step; rely on smoke tests during implementation iterations |
| 2026-03-10 | self | Tried to bind a local HTTP server inside the sandbox and hit `PermissionError: [Errno 1] Operation not permitted` | For local web UI verification in this workspace, request escalated permissions before starting the server |

## User Preferences
- Build practical local pipelines end-to-end instead of stopping at design.

## Patterns That Work
- `tesseract` is already installed globally and can OCR these screenshots well enough once the browser chrome is cropped out.
- The screenshot filenames sort into reading order because they are timestamped.
- A repo-local `.venv` is the safest place to install Python packages for this workspace.
- Projection-based page tightening removes toolbar/control OCR noise more reliably than the earlier contour-only fallback.

## Patterns That Don't Work
- Full-screenshot OCR without cropping reads `archive.org`, the toolbar, and viewer controls into the transcription.
- Mixed image/text spreads still need manual review because the current heuristics can miss some captions or extract only one of multiple photos.

## Domain Notes
- `raw_ss/` contains 167 screenshot PNGs captured from the Internet Archive 2-up viewer.
- The pages are embedded inside browser chrome with a dark viewer background, so spread detection and page splitting are mandatory before OCR.
