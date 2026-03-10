#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse


ROOT = Path(__file__).resolve().parents[1]
RAW_SS = ROOT / "raw_ss"
DEFAULT_CROP_CONFIG_PATH = ROOT / "config" / "crop_boxes.json"


def first_screenshot_path() -> Path:
    screenshots = sorted(path for path in RAW_SS.iterdir() if path.is_file())
    if not screenshots:
        raise FileNotFoundError("No screenshots found in raw_ss/")
    return screenshots[0]


def load_crop_config() -> dict | None:
    if not DEFAULT_CROP_CONFIG_PATH.exists():
        return None
    return json.loads(DEFAULT_CROP_CONFIG_PATH.read_text(encoding="utf-8"))


def validate_box(name: str, value: dict) -> None:
    for field in ("x", "y", "width", "height"):
        if field not in value or not isinstance(value[field], (int, float)):
            raise ValueError(f"Missing or invalid {name}.{field}")
    if value["width"] <= 0 or value["height"] <= 0:
        raise ValueError(f"{name} width/height must be positive")


def validate_payload(payload: dict) -> None:
    if "image_size" not in payload or not isinstance(payload["image_size"], dict):
        raise ValueError("Missing image_size")
    size = payload["image_size"]
    if not isinstance(size.get("width"), (int, float)) or not isinstance(
        size.get("height"), (int, float)
    ):
        raise ValueError("image_size.width/image_size.height are required")

    validate_box("left", payload.get("left", {}))
    validate_box("right", payload.get("right", {}))


class CropAnnotatorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/context":
            first_image = first_screenshot_path()
            payload = {
                "first_image": {
                    "name": first_image.name,
                    "path": first_image.relative_to(ROOT).as_posix(),
                    "url": "/" + quote(first_image.relative_to(ROOT).as_posix(), safe="/"),
                },
                "crop_config_path": DEFAULT_CROP_CONFIG_PATH.relative_to(ROOT).as_posix(),
                "crop_config": load_crop_config(),
            }
            self._send_json(payload)
            return

        if parsed.path == "/api/crop-config":
            self._send_json(
                {
                    "path": DEFAULT_CROP_CONFIG_PATH.relative_to(ROOT).as_posix(),
                    "crop_config": load_crop_config(),
                }
            )
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/crop-config":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            validate_payload(payload)
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            return

        DEFAULT_CROP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_CROP_CONFIG_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._send_json(
            {
                "saved": True,
                "path": DEFAULT_CROP_CONFIG_PATH.relative_to(ROOT).as_posix(),
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the manual crop annotator.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", default=8000, type=int, help="Bind port.")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), CropAnnotatorHandler)
    print(f"Crop annotator: http://{args.host}:{args.port}/annotator/index.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
