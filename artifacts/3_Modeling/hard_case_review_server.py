#!/usr/bin/env python3
"""
Tiny local server for the hard-case review app.

Why this exists:
- `python3 -m http.server` can only serve static files.
- the browser app can keep draft state locally, but it cannot persist the CSV
  back into the repo with plain AJAX unless a backend endpoint exists.

This server keeps the same lightweight local workflow while adding:
- GET /api/health
- GET /api/review-manifest/load?relative_path=...
- POST /api/review-manifest/save

It serves the repo root as static content and only allows CSV reads/writes
inside the repository directory.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PROJECT_DIR = Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the hard-case review app with CSV save/load endpoints.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface. Defaults to 127.0.0.1.")
    parser.add_argument("--port", type=int, default=8000, help="Port number. Defaults to 8000.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=PROJECT_DIR,
        help="Repo root to serve. Defaults to the current CV_Image_pose_detection project root.",
    )
    return parser


def resolve_repo_relative_csv_path(project_dir: Path, relative_path: str) -> Path:
    if not relative_path or not relative_path.strip():
        raise ValueError("relative_path is required")
    raw_path = Path(relative_path.strip())
    if raw_path.is_absolute():
        raise ValueError("relative_path must stay relative to the project root")
    resolved = (project_dir / raw_path).resolve()
    project_root = project_dir.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("relative_path escapes the project root") from exc
    if resolved.suffix.lower() != ".csv":
        raise ValueError("only .csv files may be read or written")
    return resolved


def load_csv_text(project_dir: Path, relative_path: str) -> tuple[Path, str]:
    csv_path = resolve_repo_relative_csv_path(project_dir, relative_path)
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))
    return csv_path, csv_path.read_text(encoding="utf-8")


def save_csv_text(project_dir: Path, relative_path: str, csv_text: str) -> Path:
    csv_path = resolve_repo_relative_csv_path(project_dir, relative_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(csv_text, encoding="utf-8")
    return csv_path


def normalize_api_request_path(request_path: str, project_dir: Path) -> str:
    parsed = urlparse(request_path)
    path = parsed.path or "/"
    prefix = f"/{project_dir.resolve().name}"
    if path.startswith(prefix + "/"):
        path = path[len(prefix):]
    return path


class HardCaseReviewRequestHandler(SimpleHTTPRequestHandler):
    server_version = "HardCaseReviewServer/1.0"

    def __init__(self, *args: Any, directory: str | None = None, project_dir: Path | None = None, **kwargs: Any) -> None:
        self.project_dir = (project_dir or PROJECT_DIR).resolve()
        super().__init__(*args, directory=directory or str(self.project_dir), **kwargs)

    def end_headers(self) -> None:
        parsed = urlparse(self.path)
        normalized_api_path = normalize_api_request_path(self.path, self.project_dir)
        lower_path = parsed.path.lower()
        if normalized_api_path.startswith("/api/"):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
        elif lower_path.endswith((".html", ".js", ".css", ".csv", ".json")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_OPTIONS(self) -> None:  # noqa: N802
        normalized_path = normalize_api_request_path(self.path, self.project_dir)
        if normalized_path.startswith("/api/"):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        normalized_path = normalize_api_request_path(self.path, self.project_dir)
        if normalized_path == "/api/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "project_root": str(self.project_dir),
                    "project_name": self.project_dir.name,
                    "api_base": "/api",
                    "default_manifest_path": "artifacts/3_Modeling/training_outputs/hard_case_review_manifest.csv",
                },
            )
            return
        if normalized_path == "/api/review-manifest/load":
            relative_path = parse_qs(parsed.query).get("relative_path", [""])[0]
            try:
                csv_path, csv_text = load_csv_text(self.project_dir, relative_path)
            except FileNotFoundError:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "file not found", "relative_path": relative_path})
                return
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "relative_path": relative_path,
                    "absolute_path": str(csv_path),
                    "csv_text": csv_text,
                },
            )
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        normalized_path = normalize_api_request_path(self.path, self.project_dir)
        if normalized_path != "/api/review-manifest/save":
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown API route"})
            return

        try:
            payload = self._read_json_body()
            relative_path = str(payload.get("relative_path", "")).strip()
            csv_text = payload.get("csv_text", "")
            if not isinstance(csv_text, str):
                raise ValueError("csv_text must be a string")
            saved_path = save_csv_text(self.project_dir, relative_path, csv_text)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "relative_path": relative_path,
                "absolute_path": str(saved_path),
                "bytes_written": len(csv_text.encode("utf-8")),
            },
        )


def run_server(host: str, port: int, project_dir: Path) -> None:
    handler = partial(
        HardCaseReviewRequestHandler,
        directory=str(project_dir.resolve()),
        project_dir=project_dir.resolve(),
    )
    with ThreadingHTTPServer((host, port), handler) as httpd:
        print(f"Serving {project_dir.resolve()} at http://{host}:{port}")
        print("Review app: /artifacts/3_Modeling/hard_case_review_app.html")
        print("API health: /api/health")
        httpd.serve_forever()


def main() -> None:
    args = build_arg_parser().parse_args()
    run_server(args.host, args.port, args.project_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
