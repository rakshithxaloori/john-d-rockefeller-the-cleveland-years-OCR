from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "book_pipeline.py"
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


if __name__ == "__main__":
    unittest.main()
