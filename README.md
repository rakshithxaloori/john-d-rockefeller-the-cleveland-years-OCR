# Screenshot Book Pipeline

Extracts OCR text, page numbers, and image assets from the screenshots in `raw_ss/`, then renders a page-preserving Markdown transcription.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`tesseract` must already be installed and available on `PATH`.

## Usage

```bash
.venv/bin/python scripts/book_pipeline.py extract --input raw_ss --out artifacts
.venv/bin/python scripts/book_pipeline.py render --manifest artifacts/book_manifest.json --overrides artifacts/manual_overrides.json --out output/book.md
```

## Outputs

- `artifacts/book_manifest.json`: extracted spreads, pages, text blocks, images, metadata, and issues
- `artifacts/manual_overrides.json`: review file for manual corrections before rendering
- `artifacts/pages/`: cropped page images
- `artifacts/assets/photos/`: extracted photos, maps, and illustrations
- `output/book.md`: rendered Markdown
- `output/assets/photos/`: copied image assets referenced by the Markdown

## Overrides Format

`manual_overrides.json` is keyed by `page_id`. Supported keys:

```json
{
  "0001-right": {
    "printed_page_number": "4",
    "page_type": "title",
    "text_block_updates": [
      { "index": 0, "kind": "title", "text": "JOHN D. ROCKEFELLER" }
    ],
    "image_region_updates": [
      { "index": 0, "caption": "Updated caption", "keep": true }
    ]
  }
}
```

You can also replace entire `text_blocks` or `image_regions` arrays for a page.

## Manual Crop Annotator

If the automatic left/right page split is off, you can define fixed page boxes from the first screenshot and reuse them for every screenshot in `raw_ss/`.

```bash
.venv/bin/python scripts/crop_annotator_server.py
```

Then open [http://127.0.0.1:8000/annotator/index.html](http://127.0.0.1:8000/annotator/index.html), drag a left box and a right box, and save. The annotator writes `config/crop_boxes.json`.

Once that file exists, extracting from the repo’s `raw_ss/` will automatically use it:

```bash
.venv/bin/python scripts/book_pipeline.py extract --input raw_ss --out artifacts
```

You can also point at a different crop config explicitly:

```bash
.venv/bin/python scripts/book_pipeline.py extract --input raw_ss --out artifacts --crop-config config/crop_boxes.json
```
