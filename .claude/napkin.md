# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|
| 2026-03-10 | self | Assumed `pip install --user` would work with the system Python | Use a repo-local virtualenv in this workspace; Homebrew Python is PEP 668 managed |
| 2026-03-10 | self | Used a bad `sed` expression while listing repo files | Keep file discovery commands simple here; `find` and `rg --files` are enough |
| 2026-03-10 | self | Expected the full 167-screenshot OCR pass to finish quickly enough for an in-turn verification run | Treat the full dataset pass as a separate long-running validation step; rely on smoke tests during implementation iterations |
| 2026-03-10 | self | Tried to bind a local HTTP server inside the sandbox and hit `PermissionError: [Errno 1] Operation not permitted` | For local web UI verification in this workspace, request escalated permissions before starting the server |
| 2026-03-11 | self | Used an overcomplicated `rg` pattern and got a regex parse error while tracing page-number generation | Use plain literal searches or multiple simple `rg` runs when inspecting this repo |
| 2026-03-11 | self | Imported `scripts/book_pipeline.py` in tests via `importlib` without registering the module in `sys.modules`, which broke `@dataclass` processing | When loading repo scripts by spec in tests, insert the module into `sys.modules` before `exec_module` |
| 2026-03-11 | self | Tried to update the Git index in the sandbox and hit `.git/index.lock: Operation not permitted` | For commands that modify tracked files or staging state here, request escalated permissions for the Git operation |
| 2026-03-11 | self | README had drifted from the current page-label resolver and did not mention the project's Codex-assisted origin | Keep `README.md` updated when page-number heuristics or project provenance need to be explicit |
| 2026-03-13 | self | Used a `find ... -prune -o ... -delete` expression that still tried to touch `./.git/.DS_Store` | When deleting junk files here, verify `find` prune/delete precedence carefully or target explicit paths instead |

## User Preferences
- Build practical local pipelines end-to-end instead of stopping at design.
- Keep local environment, OS metadata, and Python cache artifacts out of Git.

## Patterns That Work
- `tesseract` is already installed globally and can OCR these screenshots well enough once the browser chrome is cropped out.
- The screenshot filenames sort into reading order because they are timestamped.
- A repo-local `.venv` is the safest place to install Python packages for this workspace.
- Projection-based page tightening removes toolbar/control OCR noise more reliably than the earlier contour-only fallback.
- Persisting `trailing_page_number_candidate` in `book_manifest.json` preserves enough raw evidence to re-resolve page labels on later rerenders without re-running OCR.

## Patterns That Don't Work
- Full-screenshot OCR without cropping reads `archive.org`, the toolbar, and viewer controls into the transcription.
- Mixed image/text spreads still need manual review because the current heuristics can miss some captions or extract only one of multiple photos.

## Domain Notes
- `raw_ss/` contains 167 screenshot PNGs captured from the Internet Archive 2-up viewer.
- The pages are embedded inside browser chrome with a dark viewer background, so spread detection and page splitting are mandatory before OCR.
- `output/book.md` page headers can mix OCR-detected printed page numbers with viewer fallback labels, so header order is not guaranteed to be monotonic.
- In `output/book.md`, the common numbered-page heading format is `## Page <digits>`, but at least one front-matter heading uses a roman numeral (`## Page XI`).
- `scripts/book_pipeline.py` only OCRs digit-only printed page numbers, so roman-numeral front matter will fall back to viewer labels or produce bogus numeric matches.
- `.gitignore` contains `**__pycache__**`, which does match `__pycache__` directories here, but tracked `.pyc` files already exist under `scripts/__pycache__/` and `tests/__pycache__/`.
- The repo uses a root `.venv` for local Python work; it should stay untracked and ignored.
- `raw_ss/`, `artifacts/`, `output/`, and `config/crop_boxes.json` are intentional repo data/config, so do not blanket-ignore generated-looking pipeline directories here.
