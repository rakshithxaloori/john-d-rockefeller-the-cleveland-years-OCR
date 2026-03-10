#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TESSERACT_BIN = os.environ.get("TESSERACT_BIN", "tesseract")
DEFAULT_CROP_CONFIG_PATH = Path("config/crop_boxes.json")

VIEWER_RE = re.compile(
    r"Page\s*[-—–]?\s*([A-Za-z0-9]+)?\s*\((\d+)\s*/\s*(\d+)\)",
    re.IGNORECASE,
)
VIEWER_URL_RE = re.compile(r"/page/([A-Za-z0-9]+)/mode", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"(\d{1,4})")
NOISE_SNIPPETS = (
    "archive.org",
    "return now",
    "renews automat",
    "flip right",
    "flip righ",
)


@dataclass
class OCRWord:
    text: str
    conf: float
    bbox: tuple[int, int, int, int]
    block_num: int
    par_num: int
    line_num: int
    word_num: int


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image for {path}")
    buffer.tofile(str(path))


def clamp_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int, int] | tuple[int, int]
) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def crop_image(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = clamp_bbox(bbox, image.shape)
    return image[y : y + h, x : x + w].copy()


def pad_bbox(
    bbox: tuple[int, int, int, int],
    padding_x: int,
    padding_y: int,
    shape: tuple[int, int, int] | tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    padded = (x - padding_x, y - padding_y, w + 2 * padding_x, h + 2 * padding_y)
    return clamp_bbox(padded, shape)


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return max(0, bbox[2]) * max(0, bbox[3])


def bbox_intersection(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> int:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def bbox_horizontal_overlap_ratio(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    ax1, _, aw, _ = a
    bx1, _, bw, _ = b
    ax2 = ax1 + aw
    bx2 = bx1 + bw
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    base = max(1, min(aw, bw))
    return overlap / base


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def clean_ocr_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_block_text(words: list[OCRWord]) -> str:
    if not words:
        return ""

    grouped: dict[tuple[int, int, int], list[OCRWord]] = {}
    for word in words:
        key = (word.block_num, word.par_num, word.line_num)
        grouped.setdefault(key, []).append(word)

    lines: list[tuple[int, int, str]] = []
    for line_words in grouped.values():
        line_words.sort(key=lambda item: item.bbox[0])
        y = min(item.bbox[1] for item in line_words)
        h = max(item.bbox[3] for item in line_words)
        text = clean_ocr_text(" ".join(item.text for item in line_words if item.text))
        if text:
            lines.append((y, h, text))
    lines.sort(key=lambda item: item[0])

    paragraphs: list[str] = []
    current = ""
    previous_bottom: int | None = None
    previous_height: int | None = None

    for y, h, text in lines:
        if not current:
            current = text
        else:
            gap = y - (previous_bottom or y)
            if current.endswith("-") and text[:1].islower():
                current = current[:-1] + text
            elif gap > max(previous_height or 0, h) * 1.15:
                paragraphs.append(current.strip())
                current = text
            else:
                current = f"{current} {text}"
        previous_bottom = y + h
        previous_height = h

    if current:
        paragraphs.append(current.strip())

    return "\n\n".join(paragraphs)


def tesseract_run(
    image: np.ndarray,
    *,
    psm: int,
    output_format: str = "txt",
    extra_config: list[str] | None = None,
) -> str:
    with tempfile.TemporaryDirectory(prefix="book-pipeline-") as temp_dir:
        image_path = Path(temp_dir) / "input.png"
        save_image(image_path, image)
        command = [TESSERACT_BIN, str(image_path), "stdout", "--psm", str(psm)]
        if extra_config:
            command.extend(extra_config)
        if output_format == "tsv":
            command.append("tsv")

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Tesseract failed with code {result.returncode}: {result.stderr.strip()}"
            )
        return result.stdout


def parse_tsv(tsv_text: str) -> list[OCRWord]:
    words: list[OCRWord] = []
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        conf_text = row.get("conf") or "-1"
        try:
            conf = float(conf_text)
        except ValueError:
            conf = -1.0
        if conf < 0:
            continue

        try:
            left = int(row.get("left") or "0")
            top = int(row.get("top") or "0")
            width = int(row.get("width") or "0")
            height = int(row.get("height") or "0")
            block_num = int(row.get("block_num") or "0")
            par_num = int(row.get("par_num") or "0")
            line_num = int(row.get("line_num") or "0")
            word_num = int(row.get("word_num") or "0")
        except ValueError:
            continue

        if width <= 0 or height <= 0:
            continue

        words.append(
            OCRWord(
                text=text,
                conf=conf,
                bbox=(left, top, width, height),
                block_num=block_num,
                par_num=par_num,
                line_num=line_num,
                word_num=word_num,
            )
        )
    return words


def prepare_dark_strip_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    normalized = cv2.normalize(upscaled, None, 0, 255, cv2.NORM_MINMAX)
    _, threshold = cv2.threshold(normalized, 140, 255, cv2.THRESH_BINARY)
    inverted = 255 - threshold
    return cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)


def prepare_footer_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    normalized = cv2.normalize(upscaled, None, 0, 255, cv2.NORM_MINMAX)
    _, threshold = cv2.threshold(
        normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)


def prepare_page_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def load_crop_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    for key in ("left", "right"):
        if key not in data:
            raise ValueError(f"Crop config missing '{key}' box")
        box = data[key]
        for field in ("x", "y", "width", "height"):
            value = box.get(field)
            if not isinstance(value, (int, float)) or value <= 0 and field in {"width", "height"}:
                raise ValueError(f"Crop config '{key}.{field}' is invalid")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Crop config '{key}.{field}' is invalid")

    image_size = data.get("image_size", {})
    width = image_size.get("width")
    height = image_size.get("height")
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        raise ValueError("Crop config missing image_size.width/image_size.height")

    return data


def resolve_crop_config_path(
    candidate: Path | None, input_dir: Path | None = None
) -> Path | None:
    if candidate:
        return candidate
    default_input_dir = (Path.cwd() / "raw_ss").resolve()
    if (
        input_dir is not None
        and input_dir.resolve() == default_input_dir
        and DEFAULT_CROP_CONFIG_PATH.exists()
    ):
        return DEFAULT_CROP_CONFIG_PATH
    return None


def scale_bbox_from_crop_config(
    image_shape: tuple[int, int, int],
    crop_config: dict[str, Any],
    side: str,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    reference_size = crop_config["image_size"]
    scale_x = width / float(reference_size["width"])
    scale_y = height / float(reference_size["height"])
    source_box = crop_config[side]
    bbox = (
        int(round(source_box["x"] * scale_x)),
        int(round(source_box["y"] * scale_y)),
        int(round(source_box["width"] * scale_x)),
        int(round(source_box["height"] * scale_y)),
    )
    return clamp_bbox(bbox, image_shape)


def light_page_mask(
    image: np.ndarray,
    *,
    min_value: int = 140,
    max_saturation: int = 48,
    close_kernel: tuple[int, int] = (29, 29),
) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, min_value), (179, max_saturation, 255))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, close_kernel),
    )
    return mask


def detect_primary_bright_rect(
    image: np.ndarray,
    *,
    threshold_value: int = 176,
    min_area_ratio: float = 0.18,
) -> tuple[tuple[int, int, int, int], bool]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = image.shape[0] * image.shape[1]
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < image_area * min_area_ratio:
            continue
        candidates.append((area, (x, y, w, h)))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return clamp_bbox(candidates[0][1], image.shape), True

    fallback = (
        int(image.shape[1] * 0.16),
        int(image.shape[0] * 0.08),
        int(image.shape[1] * 0.68),
        int(image.shape[0] * 0.80),
    )
    return clamp_bbox(fallback, image.shape), False


def split_spread(spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = spread.shape[1]
    center = width // 2
    overlap = max(24, int(width * 0.012))
    left = spread[:, : min(width, center + overlap)]
    right = spread[:, max(0, center - overlap) :]
    return left, right


def tighten_page_canvas(page_image: np.ndarray) -> np.ndarray:
    height, width = page_image.shape[:2]
    mask = light_page_mask(page_image, min_value=150, close_kernel=(19, 19))
    row_ratio = np.mean(mask > 0, axis=1)
    col_ratio = np.mean(mask > 0, axis=0)
    row_indices = np.where(row_ratio > 0.18)[0]
    col_indices = np.where(col_ratio > 0.08)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return page_image

    projection_bbox = (
        int(col_indices[0]),
        int(row_indices[0]),
        int(col_indices[-1] - col_indices[0] + 1),
        int(row_indices[-1] - row_indices[0] + 1),
    )
    projection_bbox = pad_bbox(projection_bbox, 6, 6, page_image.shape)
    return crop_image(page_image, projection_bbox)


def refine_page_crop(page_half: np.ndarray) -> tuple[np.ndarray, bool]:
    height, width = page_half.shape[:2]
    mask = light_page_mask(page_half, min_value=132, close_kernel=(31, 31))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox: tuple[int, int, int, int] | None = None
    best_score = 0.0
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        bbox = (x, y, w, h)
        if contour_area < height * width * 0.12:
            continue
        if w < width * 0.45 or h < height * 0.68:
            continue
        bbox_area_value = bbox_area(bbox)
        fill_ratio = contour_area / max(1, bbox_area_value)
        score = contour_area * fill_ratio
        if score > best_score:
            best_score = score
            best_bbox = bbox

    if best_bbox is not None:
        padded = pad_bbox(best_bbox, 10, 10, page_half.shape)
        return tighten_page_canvas(crop_image(page_half, padded)), True

    fallback = (
        int(width * 0.04),
        int(height * 0.07),
        int(width * 0.90),
        int(height * 0.84),
    )
    return tighten_page_canvas(crop_image(page_half, fallback)), False


def extract_viewer_metadata(image: np.ndarray) -> dict[str, Any]:
    height, width = image.shape[:2]
    top_strip = prepare_dark_strip_for_ocr(image[:120, :])
    top_center_strip = prepare_dark_strip_for_ocr(
        image[:90, int(width * 0.16) : int(width * 0.84)]
    )
    bottom_strip = prepare_dark_strip_for_ocr(image[-140:, :])
    bottom_left_strip = prepare_dark_strip_for_ocr(
        image[int(height * 0.90) :, : int(width * 0.45)]
    )

    ocr_samples = [
        tesseract_run(top_strip, psm=6),
        tesseract_run(top_center_strip, psm=11),
        tesseract_run(bottom_strip, psm=6),
        tesseract_run(bottom_left_strip, psm=6),
    ]
    combined = "\n".join(ocr_samples)

    viewer_page_token: str | None = None
    viewer_sequence: int | None = None
    viewer_total: int | None = None

    viewer_match = VIEWER_RE.search(combined)
    if viewer_match:
        if viewer_match.group(1):
            viewer_page_token = viewer_match.group(1)
        viewer_sequence = int(viewer_match.group(2))
        viewer_total = int(viewer_match.group(3))

    if viewer_page_token is None:
        url_match = VIEWER_URL_RE.search(combined)
        if url_match:
            viewer_page_token = url_match.group(1)

    return {
        "viewer_page_token": viewer_page_token,
        "viewer_sequence": viewer_sequence,
        "viewer_total": viewer_total,
        "ocr_text": clean_ocr_text(combined),
    }


def extract_printed_page_number(page_image: np.ndarray) -> str | None:
    height, width = page_image.shape[:2]
    candidates: list[str] = []
    footer_boxes = [
        (
            int(width * 0.28),
            int(height * 0.89),
            int(width * 0.44),
            int(height * 0.11),
        ),
        (
            int(width * 0.24),
            int(height * 0.90),
            int(width * 0.52),
            int(height * 0.09),
        ),
        (
            int(width * 0.34),
            int(height * 0.87),
            int(width * 0.32),
            int(height * 0.13),
        ),
    ]

    for footer_bbox in footer_boxes:
        footer = crop_image(page_image, footer_bbox)
        prepared = prepare_footer_for_ocr(footer)
        for psm in (7, 13):
            text = tesseract_run(
                prepared,
                psm=psm,
                extra_config=["-c", "tessedit_char_whitelist=0123456789"],
            )
            matches = PAGE_NUMBER_RE.findall(text)
            candidates.extend(matches)

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-len(item), item))
    return candidates[0]


def words_confidence(words: list[OCRWord]) -> float:
    if not words:
        return 0.0
    return average([word.conf for word in words])


def make_text_seed_mask(shape: tuple[int, int], words: list[OCRWord]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for word in words:
        x, y, w, h = word.bbox
        pad_x = max(4, int(w * 0.18))
        pad_y = max(3, int(h * 0.25))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        mask[y1:y2, x1:x2] = 255

    if words:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)),
        )
        mask = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 11)),
            iterations=1,
        )
    return mask


def detect_text_block_bboxes(
    text_seed_mask: np.ndarray, page_shape: tuple[int, int, int]
) -> list[tuple[int, int, int, int]]:
    page_height, page_width = page_shape[:2]
    contours, _ = cv2.findContours(
        text_seed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < max(1800, int(page_height * page_width * 0.0015)):
            continue
        if w < page_width * 0.10 and h < page_height * 0.05:
            continue
        bboxes.append((x, y, w, h))
    bboxes.sort(key=lambda bbox: (bbox[1], bbox[0]))
    return bboxes


def merge_bboxes(
    boxes: list[tuple[int, int, int, int]], max_gap: int
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    remaining = sorted(boxes, key=lambda bbox: (bbox[1], bbox[0]))
    merged: list[tuple[int, int, int, int]] = []

    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            next_remaining: list[tuple[int, int, int, int]] = []
            for other in remaining:
                overlap = bbox_intersection(
                    pad_bbox(current, max_gap, max_gap, (1_000_000, 1_000_000)),
                    other,
                )
                same_band = abs(current[1] - other[1]) <= max_gap or abs(
                    (current[1] + current[3]) - (other[1] + other[3])
                ) <= max_gap
                if overlap > 0 or same_band:
                    x1 = min(current[0], other[0])
                    y1 = min(current[1], other[1])
                    x2 = max(current[0] + current[2], other[0] + other[2])
                    y2 = max(current[1] + current[3], other[1] + other[3])
                    current = (x1, y1, x2 - x1, y2 - y1)
                    changed = True
                else:
                    next_remaining.append(other)
            remaining = next_remaining
        merged.append(current)
    merged.sort(key=lambda bbox: (bbox[1], bbox[0]))
    return merged


def detect_image_bboxes(
    page_image: np.ndarray, text_seed_mask: np.ndarray
) -> list[tuple[int, int, int, int]]:
    height, width = page_image.shape[:2]
    page_area = height * width
    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
    ink_mask = np.where(gray < 226, 255, 0).astype(np.uint8)
    text_mask = cv2.dilate(
        text_seed_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)),
        iterations=1,
    )
    non_text = cv2.bitwise_and(ink_mask, cv2.bitwise_not(text_mask))
    non_text = cv2.morphologyEx(
        non_text,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19)),
    )
    non_text = cv2.dilate(
        non_text,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        non_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates: list[tuple[int, int, int, int]] = []
    min_area = int(height * width * 0.04)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area:
            continue
        if w < width * 0.14 or h < height * 0.10:
            continue
        bbox = pad_bbox((x, y, w, h), 8, 8, page_image.shape)
        region = crop_image(gray, bbox)
        std_dev = float(np.std(region))
        dark_ratio = float(np.mean(region < 196))
        text_overlap = float(np.mean(text_seed_mask[y : y + h, x : x + w] > 0))
        if std_dev < 18 and dark_ratio < 0.08:
            continue
        if text_overlap > 0.40:
            continue
        if (
            bbox_area(bbox) > page_area * 0.70
            and text_overlap > 0.05
            and dark_ratio < 0.20
        ):
            continue
        if (
            x <= 12
            and y <= 12
            and (x + w) >= width - 12
            and (y + h) >= height - 12
            and text_overlap > 0.03
        ):
            continue
        candidates.append(bbox)

    candidates = merge_bboxes(candidates, max_gap=16)
    filtered: list[tuple[int, int, int, int]] = []
    for bbox in candidates:
        area = bbox_area(bbox)
        if area < min_area:
            continue
        filtered.append(bbox)
    return filtered


def select_block_psm(
    bbox: tuple[int, int, int, int],
    words_in_bbox: list[OCRWord],
    text_seed_mask: np.ndarray,
) -> int:
    x, y, w, h = bbox
    density = float(np.mean(text_seed_mask[y : y + h, x : x + w] > 0))
    if len(words_in_bbox) >= 30 or density > 0.12 or h > 220:
        return 6
    return 11


def page_has_prose(text_blocks: list[dict[str, Any]]) -> bool:
    return any(block["kind"] in {"body", "quote"} for block in text_blocks)


def guess_text_block_kind(
    *,
    bbox: tuple[int, int, int, int],
    text: str,
    page_shape: tuple[int, int, int],
    image_bboxes: list[tuple[int, int, int, int]],
    median_word_height: float,
    average_word_height: float,
) -> str:
    height, width = page_shape[:2]
    x, y, w, h = bbox
    center_x = x + (w / 2)
    text_length = len(text)

    for image_bbox in image_bboxes:
        image_x, image_y, image_w, image_h = image_bbox
        image_bottom = image_y + image_h
        gap_below = y - image_bottom
        if (
            gap_below >= -10
            and gap_below <= max(70, int(height * 0.05))
            and bbox_horizontal_overlap_ratio(bbox, image_bbox) >= 0.45
            and text_length <= 420
        ):
            return "caption"

    is_centered = abs(center_x - (width / 2)) <= width * 0.11
    if (
        y <= height * 0.30
        and text_length <= 220
        and is_centered
        and average_word_height >= max(median_word_height * 1.15, 18)
    ):
        return "title"

    stripped = text.lstrip()
    if stripped.startswith(("\"", "'", "“")):
        return "quote"

    if (
        w <= width * 0.76
        and x >= width * 0.08
        and (x + w) <= width * 0.92
        and text_length <= 650
    ):
        return "quote"

    return "body"


def determine_page_type(
    text_blocks: list[dict[str, Any]],
    image_regions: list[dict[str, Any]],
) -> str:
    if not text_blocks and not image_regions:
        return "blank"
    if image_regions and not text_blocks:
        return "illustration"
    if any(block["kind"] == "title" for block in text_blocks) and len(text_blocks) <= 4:
        return "title"
    if image_regions and any(block["kind"] == "body" for block in text_blocks):
        return "mixed"
    return "body"


def build_text_blocks(
    page_image: np.ndarray,
    text_seed_mask: np.ndarray,
    initial_words: list[OCRWord],
    image_bboxes: list[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    candidate_bboxes = detect_text_block_bboxes(text_seed_mask, page_image.shape)
    initial_word_heights = [word.bbox[3] for word in initial_words]
    page_median_word_height = median(initial_word_heights, default=18.0)
    blocks: list[dict[str, Any]] = []

    for bbox in candidate_bboxes:
        if any(
            bbox_intersection(bbox, image_bbox) / max(1, bbox_area(bbox)) > 0.30
            for image_bbox in image_bboxes
        ):
            continue

        x, y, w, h = bbox
        crop = crop_image(page_image, pad_bbox(bbox, 10, 10, page_image.shape))
        words_in_bbox = [
            word
            for word in initial_words
            if bbox_intersection(bbox, word.bbox) > 0
        ]
        psm = select_block_psm(bbox, words_in_bbox, text_seed_mask)
        prepared_crop = prepare_page_for_ocr(crop)
        ocr_words = parse_tsv(tesseract_run(prepared_crop, psm=psm, output_format="tsv"))
        if not ocr_words:
            fallback_psm = 11 if psm == 6 else 6
            ocr_words = parse_tsv(
                tesseract_run(prepared_crop, psm=fallback_psm, output_format="tsv")
            )
        if not ocr_words:
            continue

        block_text = normalize_block_text(ocr_words)
        if not block_text:
            continue
        if any(snippet in block_text.lower() for snippet in NOISE_SNIPPETS):
            continue

        if y >= page_image.shape[0] * 0.86 and re.fullmatch(r"\d{1,4}", block_text):
            continue

        average_height = average([word.bbox[3] for word in ocr_words])
        kind = guess_text_block_kind(
            bbox=bbox,
            text=block_text,
            page_shape=page_image.shape,
            image_bboxes=image_bboxes,
            median_word_height=page_median_word_height,
            average_word_height=average_height,
        )

        blocks.append(
            {
                "kind": kind,
                "text": block_text,
                "bbox": [x, y, w, h],
                "confidence": round(words_confidence(ocr_words), 2),
                "keep": True,
            }
        )

    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return blocks


def attach_captions_to_images(
    text_blocks: list[dict[str, Any]], image_regions: list[dict[str, Any]]
) -> None:
    for image_region in image_regions:
        image_bbox = tuple(image_region["bbox"])
        best_index: int | None = None
        best_gap = sys.maxsize

        for index, block in enumerate(text_blocks):
            if block["kind"] != "caption":
                continue
            bbox = tuple(block["bbox"])
            gap = bbox[1] - (image_bbox[1] + image_bbox[3])
            overlap = bbox_horizontal_overlap_ratio(bbox, image_bbox)
            if gap < -10 or overlap < 0.45:
                continue
            if gap < best_gap:
                best_gap = gap
                best_index = index

        if best_index is not None:
            image_region["caption"] = text_blocks[best_index]["text"]
            image_region["caption_block_index"] = best_index


def extract_image_regions(
    *,
    page_image: np.ndarray,
    image_bboxes: list[tuple[int, int, int, int]],
    page_id: str,
    assets_dir: Path,
    text_seed_mask: np.ndarray,
) -> list[dict[str, Any]]:
    image_regions: list[dict[str, Any]] = []
    for index, bbox in enumerate(image_bboxes, start=1):
        x, y, w, h = bbox
        crop = crop_image(page_image, bbox)
        asset_path = assets_dir / f"{page_id}-{index:02d}.png"
        save_image(asset_path, crop)
        overlap_ratio = float(np.mean(text_seed_mask[y : y + h, x : x + w] > 0))
        image_regions.append(
            {
                "asset_path": repo_relative(asset_path),
                "bbox": [x, y, w, h],
                "caption": None,
                "confidence": round((1.0 - min(overlap_ratio, 0.95)) * 100.0, 2),
                "keep": True,
            }
        )
    return image_regions


def strip_trailing_page_number(
    text_blocks: list[dict[str, Any]], printed_page_number: str | None
) -> None:
    if not printed_page_number:
        return
    pattern = re.compile(rf"(?:\s+|\n+){re.escape(printed_page_number)}\s*$")
    for block in text_blocks:
        block["text"] = pattern.sub("", block["text"]).strip()


def extract_page(
    *,
    page_image: np.ndarray,
    page_id: str,
    side: str,
    viewer_metadata: dict[str, Any],
    pages_dir: Path,
    assets_dir: Path,
    crop_confident: bool,
    crop_mode: str,
) -> dict[str, Any]:
    page_crop_path = pages_dir / f"{page_id}.png"
    save_image(page_crop_path, page_image)

    printed_page_number = extract_printed_page_number(page_image)
    initial_words = parse_tsv(
        tesseract_run(prepare_page_for_ocr(page_image), psm=11, output_format="tsv")
    )
    text_seed_mask = make_text_seed_mask(page_image.shape[:2], initial_words)
    image_bboxes = detect_image_bboxes(page_image, text_seed_mask)
    text_blocks = build_text_blocks(page_image, text_seed_mask, initial_words, image_bboxes)
    strip_trailing_page_number(text_blocks, printed_page_number)
    image_regions = extract_image_regions(
        page_image=page_image,
        image_bboxes=image_bboxes,
        page_id=page_id,
        assets_dir=assets_dir,
        text_seed_mask=text_seed_mask,
    )
    attach_captions_to_images(text_blocks, image_regions)
    page_type = determine_page_type(text_blocks, image_regions)

    issues: list[str] = []
    weighted_confidences = [block["confidence"] for block in text_blocks if block["keep"]]
    average_confidence = round(average(weighted_confidences), 2) if weighted_confidences else 0.0

    if text_blocks and average_confidence < 80.0:
        issues.append(f"low_ocr_confidence:{average_confidence}")
    if page_has_prose(text_blocks) and not printed_page_number:
        issues.append("missing_printed_page_number")
    if not crop_confident:
        issues.append("page_crop_low_confidence")

    for image_region in image_regions:
        bbox = tuple(image_region["bbox"])
        x, y, w, h = bbox
        overlap_ratio = float(np.mean(text_seed_mask[y : y + h, x : x + w] > 0))
        if overlap_ratio > 0.10:
            issues.append(f"image_text_overlap:{overlap_ratio:.3f}")

    return {
        "page_id": page_id,
        "side": side,
        "printed_page_number": printed_page_number,
        "page_type": page_type,
        "crop_path": repo_relative(page_crop_path),
        "crop_mode": crop_mode,
        "text_blocks": text_blocks,
        "image_regions": image_regions,
        "issues": issues,
        "average_ocr_confidence": average_confidence,
        "viewer_page_token": viewer_metadata.get("viewer_page_token"),
        "viewer_sequence": viewer_metadata.get("viewer_sequence"),
        "viewer_total": viewer_metadata.get("viewer_total"),
    }


def extract_spread(
    *,
    source_path: Path,
    spread_index: int,
    pages_dir: Path,
    assets_dir: Path,
    crop_config: dict[str, Any] | None,
) -> dict[str, Any]:
    image = read_image(source_path)
    viewer_metadata = extract_viewer_metadata(image)
    crop_mode = "auto"

    if crop_config:
        crop_mode = "manual"
        left_page = crop_image(image, scale_bbox_from_crop_config(image.shape, crop_config, "left"))
        right_page = crop_image(image, scale_bbox_from_crop_config(image.shape, crop_config, "right"))
        spread_confident = True
        left_confident = True
        right_confident = True
    else:
        spread_bbox, spread_confident = detect_primary_bright_rect(image)
        spread = crop_image(image, spread_bbox)
        left_half, right_half = split_spread(spread)
        left_page, left_confident = refine_page_crop(left_half)
        right_page, right_confident = refine_page_crop(right_half)

    spread_id = f"{spread_index:04d}"
    pages = [
        extract_page(
            page_image=left_page,
            page_id=f"{spread_id}-left",
            side="left",
            viewer_metadata=viewer_metadata,
            pages_dir=pages_dir,
            assets_dir=assets_dir,
            crop_confident=spread_confident and left_confident,
            crop_mode=crop_mode,
        ),
        extract_page(
            page_image=right_page,
            page_id=f"{spread_id}-right",
            side="right",
            viewer_metadata=viewer_metadata,
            pages_dir=pages_dir,
            assets_dir=assets_dir,
            crop_confident=spread_confident and right_confident,
            crop_mode=crop_mode,
        ),
    ]

    if not spread_confident:
        for page in pages:
            page["issues"].append("spread_crop_low_confidence")

    return {
        "spread_id": spread_id,
        "source_path": repo_relative(source_path),
        "viewer_page_token": viewer_metadata.get("viewer_page_token"),
        "viewer_sequence": viewer_metadata.get("viewer_sequence"),
        "viewer_total": viewer_metadata.get("viewer_total"),
        "pages": pages,
    }


def load_overrides(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_updates(items: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patched = [dict(item) for item in items]
    for update in updates:
        index = update.get("index")
        if not isinstance(index, int) or index < 0 or index >= len(patched):
            continue
        merged = dict(patched[index])
        for key, value in update.items():
            if key != "index":
                merged[key] = value
        if merged.get("keep", True):
            patched[index] = merged
        else:
            patched[index] = merged
    return patched


def apply_page_override(page: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    patched = dict(page)
    if "printed_page_number" in override:
        patched["printed_page_number"] = override["printed_page_number"]
    if "page_type" in override:
        patched["page_type"] = override["page_type"]
    if "issues" in override:
        patched["issues"] = override["issues"]

    text_blocks = [dict(block) for block in page.get("text_blocks", [])]
    if "text_blocks" in override:
        text_blocks = override["text_blocks"]
    elif "text_block_updates" in override:
        text_blocks = apply_updates(text_blocks, override["text_block_updates"])

    image_regions = [dict(region) for region in page.get("image_regions", [])]
    if "image_regions" in override:
        image_regions = override["image_regions"]
    elif "image_region_updates" in override:
        image_regions = apply_updates(image_regions, override["image_region_updates"])

    patched["text_blocks"] = text_blocks
    patched["image_regions"] = image_regions
    return patched


def copy_render_asset(
    source_path: Path, output_assets_dir: Path, output_root: Path
) -> str:
    destination = output_assets_dir / source_path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return destination.relative_to(output_root).as_posix()


def markdown_for_text_block(block: dict[str, Any]) -> str:
    text = block["text"].strip()
    if not text:
        return ""
    kind = block.get("kind")
    if kind == "title":
        return f"### {text}"
    if kind == "quote":
        paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
        return "\n\n".join(f"> {paragraph.replace(chr(10), ' ')}" for paragraph in paragraphs)
    return text


def render_markdown(
    manifest_path: Path,
    overrides_path: Path,
    output_path: Path,
) -> None:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    overrides = load_overrides(overrides_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_assets_dir = output_path.parent / "assets" / "photos"

    spreads = manifest.get("spreads", [])
    spreads.sort(
        key=lambda item: (
            item.get("viewer_sequence") is None,
            item.get("viewer_sequence") or sys.maxsize,
            item.get("spread_id"),
        )
    )

    chunks: list[str] = []
    for spread in spreads:
        pages = spread.get("pages", [])
        pages.sort(key=lambda page: 0 if page.get("side") == "left" else 1)
        for page in pages:
            override = overrides.get(page["page_id"], {})
            rendered_page = apply_page_override(page, override)

            printed_page_number = rendered_page.get("printed_page_number")
            viewer_page_token = rendered_page.get("viewer_page_token") or spread.get(
                "viewer_page_token"
            )
            viewer_sequence = rendered_page.get("viewer_sequence") or spread.get(
                "viewer_sequence"
            )
            viewer_total = rendered_page.get("viewer_total") or spread.get("viewer_total")

            heading = (
                f"## Page {printed_page_number}"
                if printed_page_number
                else f"## Viewer Page {viewer_page_token or viewer_sequence or 'unknown'}"
            )
            chunks.append(
                "\n".join(
                    [
                        (
                            f"<!-- page_id: {rendered_page['page_id']} | "
                            f"printed: {printed_page_number or 'none'} | "
                            f"viewer_token: {viewer_page_token or 'none'} | "
                            f"viewer_sequence: {viewer_sequence or 'none'}/{viewer_total or 'none'} -->"
                        ),
                        heading,
                    ]
                )
            )

            text_blocks = [
                block for block in rendered_page.get("text_blocks", []) if block.get("keep", True)
            ]
            image_regions = [
                region
                for region in rendered_page.get("image_regions", [])
                if region.get("keep", True)
            ]

            caption_block_indexes = {
                region.get("caption_block_index")
                for region in image_regions
                if region.get("caption_block_index") is not None
            }

            ordered_items: list[tuple[int, str, dict[str, Any]]] = []
            for index, block in enumerate(text_blocks):
                if index in caption_block_indexes and block.get("kind") == "caption":
                    continue
                ordered_items.append((block["bbox"][1], "text", block))
            for region in image_regions:
                ordered_items.append((region["bbox"][1], "image", region))
            ordered_items.sort(key=lambda item: item[0])

            for _, item_type, item in ordered_items:
                if item_type == "text":
                    markdown = markdown_for_text_block(item)
                    if markdown:
                        chunks.append(markdown)
                else:
                    source_path = Path(item["asset_path"])
                    if not source_path.is_absolute():
                        source_path = Path.cwd() / source_path
                    relative_asset = copy_render_asset(
                        source_path, output_assets_dir, output_path.parent
                    )
                    alt_text = f"{rendered_page['page_id']} image"
                    chunks.append(f"![{alt_text}]({relative_asset})")
                    if item.get("caption"):
                        chunks.append(f"*{item['caption'].strip()}*")

            if rendered_page.get("issues"):
                issue_text = ", ".join(rendered_page["issues"])
                chunks.append(f"<!-- issues: {issue_text} -->")

    output_path.write_text("\n\n".join(chunk for chunk in chunks if chunk.strip()) + "\n", encoding="utf-8")


def extract_command(
    input_dir: Path, output_dir: Path, crop_config_path: Path | None = None
) -> None:
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = output_dir / "pages"
    assets_dir = output_dir / "assets" / "photos"
    pages_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    screenshot_paths = sorted(path for path in input_dir.iterdir() if path.is_file())
    spreads: list[dict[str, Any]] = []
    crop_config_path = resolve_crop_config_path(crop_config_path, input_dir)
    crop_config = load_crop_config(crop_config_path) if crop_config_path else None
    if crop_config_path:
        print(f"[extract] using crop config {crop_config_path}", file=sys.stderr)

    for index, screenshot_path in enumerate(screenshot_paths, start=1):
        print(
            f"[extract] {index}/{len(screenshot_paths)} {screenshot_path.name}",
            file=sys.stderr,
        )
        spread = extract_spread(
            source_path=screenshot_path,
            spread_index=index,
            pages_dir=pages_dir,
            assets_dir=assets_dir,
            crop_config=crop_config,
        )
        spreads.append(spread)

    manifest = {
        "book": {
            "source_dir": repo_relative(input_dir),
            "page_order": "filename_timestamp_then_left_to_right",
            "crop_config_path": repo_relative(crop_config_path) if crop_config_path else None,
        },
        "spreads": spreads,
    }

    manifest_path = output_dir / "book_manifest.json"
    overrides_path = output_dir / "manual_overrides.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not overrides_path.exists():
        overrides_path.write_text("{}\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract OCR text and images from book screenshots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract OCR text, page numbers, and images.")
    extract_parser.add_argument("--input", required=True, type=Path, help="Screenshot directory.")
    extract_parser.add_argument("--out", required=True, type=Path, help="Artifacts output directory.")
    extract_parser.add_argument(
        "--crop-config",
        type=Path,
        default=None,
        help="Optional manual crop box JSON. Defaults to config/crop_boxes.json if it exists.",
    )

    render_parser = subparsers.add_parser("render", help="Render Markdown from the extracted manifest.")
    render_parser.add_argument("--manifest", required=True, type=Path, help="Path to book_manifest.json.")
    render_parser.add_argument("--overrides", required=True, type=Path, help="Path to manual_overrides.json.")
    render_parser.add_argument("--out", required=True, type=Path, help="Markdown output path.")

    args = parser.parse_args(argv)

    if args.command == "extract":
        extract_command(args.input, args.out, args.crop_config)
        return 0
    if args.command == "render":
        render_markdown(args.manifest, args.overrides, args.out)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
