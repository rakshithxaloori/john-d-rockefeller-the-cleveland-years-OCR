from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "book_pipeline.py"
MODULE_SPEC = importlib.util.spec_from_file_location("book_pipeline", SCRIPT_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
book_pipeline = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = book_pipeline
MODULE_SPEC.loader.exec_module(book_pipeline)
RAW_SS = REPO_ROOT / "raw_ss"
SMOKE_FILES = [
    "Screenshot 2026-03-10 at 12.22.14 PM.png",
    "Screenshot 2026-03-10 at 12.25.40 PM.png",
    "Screenshot 2026-03-10 at 12.30.32 PM.png",
]
NOISE_SNIPPETS = ("archive.org", "return now", "flip right", "flip righ")


class BookPipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory(prefix="book-pipeline-tests-")
        cls.temp_path = Path(cls.temp_dir.name)
        cls.input_dir = cls.temp_path / "raw"
        cls.artifacts_dir = cls.temp_path / "artifacts"
        cls.output_dir = cls.temp_path / "output"
        cls.input_dir.mkdir(parents=True, exist_ok=True)

        for name in SMOKE_FILES:
            shutil.copy2(RAW_SS / name, cls.input_dir / name)

        subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(SCRIPT_PATH),
                "extract",
                "--input",
                str(cls.input_dir),
                "--out",
                str(cls.artifacts_dir),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(SCRIPT_PATH),
                "render",
                "--manifest",
                str(cls.artifacts_dir / "book_manifest.json"),
                "--overrides",
                str(cls.artifacts_dir / "manual_overrides.json"),
                "--out",
                str(cls.output_dir / "book.md"),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

        cls.manifest = json.loads((cls.artifacts_dir / "book_manifest.json").read_text())
        cls.markdown = (cls.output_dir / "book.md").read_text()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_extract_creates_manifest_pages_and_assets(self) -> None:
        spreads = self.manifest["spreads"]
        self.assertEqual(len(spreads), 3)

        all_pages = []
        image_assets = []
        for spread in spreads:
            self.assertEqual(len(spread["pages"]), 2)
            for page in spread["pages"]:
                all_pages.append(page)
                self.assertTrue((REPO_ROOT / page["crop_path"]).exists() or Path(page["crop_path"]).exists())
                for image in page["image_regions"]:
                    image_assets.append(image["asset_path"])

        self.assertGreaterEqual(len(image_assets), 1)
        for asset in image_assets:
            self.assertTrue((REPO_ROOT / asset).exists() or Path(asset).exists())

    def test_prose_pages_extract_page_numbers_and_drop_viewer_text(self) -> None:
        pages_by_number = {}
        for spread in self.manifest["spreads"]:
            for page in spread["pages"]:
                if page["printed_page_number"]:
                    pages_by_number[page["printed_page_number"]] = page

        self.assertIn("218", pages_by_number)
        self.assertIn("219", pages_by_number)

        for number in ("218", "219"):
            page = pages_by_number[number]
            text_blob = " ".join(block["text"] for block in page["text_blocks"]).lower()
            for snippet in NOISE_SNIPPETS:
                self.assertNotIn(snippet, text_blob)

    def test_render_outputs_markdown_with_resolved_asset_links(self) -> None:
        self.assertIn("## Page 218", self.markdown)
        self.assertIn("## Page 219", self.markdown)

        for snippet in NOISE_SNIPPETS:
            self.assertNotIn(snippet, self.markdown.lower())

        for line in self.markdown.splitlines():
            if not line.startswith("!["):
                continue
            target = line.split("](", 1)[1].rstrip(")")
            self.assertTrue((self.output_dir / target).exists(), msg=target)

    def test_manual_crop_config_is_applied_exactly(self) -> None:
        config_path = self.temp_path / "crop_boxes.json"
        config_path.write_text(
            json.dumps(
                {
                    "source_image": SMOKE_FILES[0],
                    "image_size": {"width": 2880, "height": 1800},
                    "left": {"x": 180, "y": 120, "width": 640, "height": 720},
                    "right": {"x": 980, "y": 120, "width": 700, "height": 720},
                }
            ),
            encoding="utf-8",
        )

        manual_artifacts = self.temp_path / "manual-artifacts"
        subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(SCRIPT_PATH),
                "extract",
                "--input",
                str(self.input_dir),
                "--out",
                str(manual_artifacts),
                "--crop-config",
                str(config_path),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

        manual_manifest = json.loads((manual_artifacts / "book_manifest.json").read_text())
        first_spread = manual_manifest["spreads"][0]
        self.assertEqual(first_spread["pages"][0]["crop_mode"], "manual")
        self.assertEqual(first_spread["pages"][1]["crop_mode"], "manual")

        left_image = cv2.imread(str(manual_artifacts / "pages" / "0001-left.png"))
        right_image = cv2.imread(str(manual_artifacts / "pages" / "0001-right.png"))
        self.assertIsNotNone(left_image)
        self.assertIsNotNone(right_image)
        self.assertEqual(left_image.shape[1], 640)
        self.assertEqual(left_image.shape[0], 720)
        self.assertEqual(right_image.shape[1], 700)
        self.assertEqual(right_image.shape[0], 720)


class PageLabelResolutionTest(unittest.TestCase):
    def test_resolver_prefers_viewer_labels_and_neighbor_consistency(self) -> None:
        manifest = {
            "spreads": [
                {
                    "spread_id": "0001",
                    "viewer_sequence": 10,
                    "viewer_total": 340,
                    "viewer_page_token": None,
                    "pages": [
                        {
                            "page_id": "0001-left",
                            "side": "left",
                            "printed_page_number": None,
                            "text_blocks": [{"text": "Index page 263", "keep": True}],
                        },
                        {
                            "page_id": "0001-right",
                            "side": "right",
                            "printed_page_number": None,
                            "text_blocks": [{"text": "Preface xi", "keep": True}],
                        },
                    ],
                },
                {
                    "spread_id": "0002",
                    "viewer_sequence": 20,
                    "viewer_total": 340,
                    "viewer_page_token": "4",
                    "pages": [
                        {
                            "page_id": "0002-left",
                            "side": "left",
                            "printed_page_number": "1",
                            "text_blocks": [{"text": "Chapter text 4", "keep": True}],
                        },
                        {
                            "page_id": "0002-right",
                            "side": "right",
                            "printed_page_number": None,
                            "text_blocks": [{"text": "More text 5", "keep": True}],
                        },
                    ],
                },
                {
                    "spread_id": "0003",
                    "viewer_sequence": 74,
                    "viewer_total": 340,
                    "viewer_page_token": None,
                    "pages": [
                        {
                            "page_id": "0003-left",
                            "side": "left",
                            "printed_page_number": None,
                            "text_blocks": [{"text": "Illustration plate", "keep": True}],
                        },
                        {
                            "page_id": "0003-right",
                            "side": "right",
                            "printed_page_number": None,
                            "text_blocks": [{"text": "Oil text 51", "keep": True}],
                        },
                    ],
                },
                {
                    "spread_id": "0004",
                    "viewer_sequence": 76,
                    "viewer_total": 340,
                    "viewer_page_token": "52",
                    "pages": [
                        {
                            "page_id": "0004-left",
                            "side": "left",
                            "printed_page_number": None,
                            "text_blocks": [{"text": "Next text 52", "keep": True}],
                        },
                        {
                            "page_id": "0004-right",
                            "side": "right",
                            "printed_page_number": None,
                            "text_blocks": [{"text": "Next text 53", "keep": True}],
                        },
                    ],
                },
            ]
        }

        book_pipeline.resolve_manifest_page_numbers(manifest)

        first_spread_pages = manifest["spreads"][0]["pages"]
        self.assertIsNone(first_spread_pages[0]["printed_page_number"])
        self.assertEqual(first_spread_pages[1]["printed_page_number"], "XI")

        second_spread_pages = manifest["spreads"][1]["pages"]
        self.assertEqual(second_spread_pages[0]["printed_page_number"], "4")
        self.assertEqual(second_spread_pages[1]["printed_page_number"], "5")
        self.assertEqual(second_spread_pages[0]["text_blocks"][0]["text"], "Chapter text")
        self.assertEqual(second_spread_pages[1]["text_blocks"][0]["text"], "More text")

        third_spread_pages = manifest["spreads"][2]["pages"]
        self.assertIsNone(third_spread_pages[0]["printed_page_number"])
        self.assertEqual(third_spread_pages[1]["printed_page_number"], "51")
        self.assertEqual(third_spread_pages[1]["text_blocks"][0]["text"], "Oil text")

    def test_render_uses_unnumbered_heading_when_no_label_is_resolved(self) -> None:
        manifest = {
            "spreads": [
                {
                    "spread_id": "0001",
                    "viewer_sequence": 4,
                    "viewer_total": 340,
                    "viewer_page_token": None,
                    "pages": [
                        {
                            "page_id": "0001-left",
                            "side": "left",
                            "printed_page_number": None,
                            "text_blocks": [{"text": "Title page 1972", "bbox": [0, 0, 1, 1], "keep": True}],
                            "image_regions": [],
                            "issues": [],
                        },
                        {
                            "page_id": "0001-right",
                            "side": "right",
                            "printed_page_number": None,
                            "text_blocks": [{"text": "More title text", "bbox": [0, 0, 1, 1], "keep": True}],
                            "image_regions": [],
                            "issues": [],
                        },
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory(prefix="book-render-test-") as temp_dir:
            temp_path = Path(temp_dir)
            manifest_path = temp_path / "book_manifest.json"
            overrides_path = temp_path / "manual_overrides.json"
            output_path = temp_path / "book.md"

            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            overrides_path.write_text("{}\n", encoding="utf-8")

            book_pipeline.render_markdown(manifest_path, overrides_path, output_path)

            markdown = output_path.read_text(encoding="utf-8")

        self.assertIn("## Unnumbered Page", markdown)
        self.assertNotIn("## Viewer Page", markdown)


if __name__ == "__main__":
    unittest.main()
