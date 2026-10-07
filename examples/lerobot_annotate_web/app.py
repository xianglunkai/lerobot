#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Review, edit, and launch official LeRobot language annotation.

    python examples/lerobot_annotate_web/app.py

Open http://127.0.0.1:8766
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import parse_qs, unquote, urlparse

from review import (
    dataset_overview,
    load_episode,
    parquet_page,
    parse_annotate_log,
    save_episode,
    staging_progress,
)
from run_pipeline import feature_plan

HOST = "127.0.0.1"
PORT = 8766
_STATIC = Path(__file__).resolve().parent / "static"
_PIPELINE = Path(__file__).resolve().parent / "run_pipeline.py"
_LOCK = threading.Lock()
_DATASET: Path | None = None
_JOB: dict[str, Any] | None = None
_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._file(_STATIC / "index.html")
            return
        if parsed.path in {"/guide", "/guide.html"}:
            self._file(_STATIC / "guide.html")
            return
        if parsed.path.startswith("/assets/"):
            self._asset(unquote(parsed.path.removeprefix("/assets/")))
            return
        try:
            if parsed.path == "/api/health":
                self._json({"status": "ok", "service": "lerobot-annotate-web"})
                return
            if parsed.path == "/api/dataset":
                self._json(dataset_overview(_require_dataset()))
                return
            if parsed.path == "/api/episode":
                episode = int(parse_qs(parsed.query).get("index", ["0"])[0])
                self._json(load_episode(_require_dataset(), episode))
                return
            if parsed.path == "/api/parquet":
                query = parse_qs(parsed.query)
                episode_text = query.get("episode", ["all"])[0]
                episode = None if episode_text in {"", "all"} else int(episode_text)
                self._json(
                    parquet_page(
                        _require_dataset(),
                        query.get("path", [""])[0],
                        offset=int(query.get("offset", ["0"])[0]),
                        limit=int(query.get("limit", ["25"])[0]),
                        episode_index=episode,
                    )
                )
                return
            if parsed.path == "/api/job":
                self._json(self._job_status())
                return
            if parsed.path.startswith("/videos/"):
                self._video(unquote(parsed.path.removeprefix("/videos/")))
                return
        except ConnectionError:
            return
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, status=400)
            return
        self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
            return
        try:
            if parsed.path == "/api/open":
                self._open(Path(str(payload["dataset_path"])))
                return
            if parsed.path == "/api/episode":
                save_episode(
                    _require_dataset(),
                    int(payload["episode_index"]),
                    list(payload.get("persistent") or []),
                    list(payload.get("events") or []),
                )
                self._json({"ok": True, "episode_index": int(payload["episode_index"])})
                return
            if parsed.path == "/api/annotate":
                self._start_job(
                    str(payload.get("episodes") or "all"),
                    str(payload.get("mode") or "overwrite"),
                    str(payload.get("output") or ""),
                    _feature_spec(payload),
                )
                return
            if parsed.path == "/api/job/control":
                self._control(str(payload.get("action") or ""))
                return
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, status=400)
            return
        self._json({"error": "not found"}, status=404)

    def _open(self, root: Path) -> None:
        global _DATASET
        root = root.expanduser().resolve()
        if not (root / "meta" / "info.json").is_file():
            raise ValueError(f"{root} is not a LeRobot dataset")
        _DATASET = root
        self._json(dataset_overview(root))

    def _start_job(self, episodes: str, mode: str, output: str, features: str) -> None:
        global _JOB
        root = _require_dataset()
        if mode not in {"overwrite", "copy"}:
            raise ValueError("mode must be overwrite or copy")
        features = feature_plan(features).spec()
        with _LOCK:
            if _JOB is not None and _JOB["process"].poll() is None:
                raise RuntimeError("an annotation job is already running")
            external = _pipeline_pid()
            if external is not None:
                raise RuntimeError(f"annotation is already running as pid {external}")
            log_path = root / ".annotate_staging" / "web_run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(_PIPELINE),
                "--root",
                str(root),
                "--episodes",
                episodes,
                "--features",
                features,
                "--mode",
                mode,
            ]
            if mode == "copy":
                command.extend(["--output", output])
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)
            _JOB = {
                "process": process,
                "log": log_path,
                "episodes": episodes,
                "mode": mode,
                "output": output,
                "features": features,
            }
        self._json(
            {"started": True, "episodes": episodes, "mode": mode, "features": features, "pid": process.pid}
        )

    def _control(self, action: str) -> None:
        pid = _pipeline_pid()
        if action not in {"pause", "resume", "stop"}:
            raise ValueError("action must be pause, resume, or stop")
        if pid is None:
            raise RuntimeError("没有正在运行的标注")
        state = _process_state(pid)
        if action == "pause":
            if state == "T":
                raise RuntimeError("标注已经暂停")
            os.kill(pid, signal.SIGSTOP)
        elif action == "resume":
            if state != "T":
                raise RuntimeError("标注没有暂停")
            os.kill(pid, signal.SIGCONT)
        else:
            if state == "T":
                os.kill(pid, signal.SIGCONT)
            os.kill(pid, signal.SIGTERM)
        self._json(self._job_status())

    def _job_status(self) -> dict[str, Any]:
        root = _DATASET
        with _LOCK:
            job = _JOB
        web_pid = None
        code = None
        text = ""
        episodes = ""
        mode = ""
        output = ""
        features = ""
        if job is not None:
            process = job["process"]
            web_pid = process.pid
            code = process.poll()
            episodes = job["episodes"]
            mode = job["mode"]
            output = job["output"]
            features = job.get("features") or ""
            log_path: Path = job["log"]
            if log_path.is_file():
                text = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        found = _pipeline_pid()
        state = _process_state(found) if found is not None else None
        paused = state == "T"
        external = found is not None and found != web_pid
        alive = found is not None and state not in {None, "Z"}
        running = alive and not paused
        return {
            "running": running,
            "paused": paused,
            "alive": alive,
            "external": external,
            "pid": found or web_pid,
            "state": state,
            "returncode": code,
            "episodes": episodes,
            "mode": mode,
            "output": output,
            "features": features,
            "log": text,
            "progress": parse_annotate_log(text),
            "staging": staging_progress(root) if root is not None else None,
        }

    def _video(self, relative: str) -> None:
        root = _require_dataset()
        requested = Path(relative)
        if requested.is_absolute() or ".." in requested.parts:
            self._json({"error": "video not found"}, status=404)
            return
        path = root / requested
        if not path.is_file():
            self._json({"error": "video not found"}, status=404)
            return
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if range_header and range_header.startswith("bytes="):
            spec = range_header.removeprefix("bytes=")
            start_s, _, end_s = spec.partition("-")
            if start_s:
                start = int(start_s)
            if end_s:
                end = int(end_s)
            end = min(end, size - 1)
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        try:
            self.end_headers()
        except ConnectionError:
            return
        with path.open("rb") as handle:
            stream_file(handle, self.wfile.write, start, length)

    def _asset(self, relative: str) -> None:
        path = (_STATIC / relative).resolve()
        if not path.is_relative_to(_STATIC.resolve()) or not path.is_file():
            self._json({"error": "not found"}, status=404)
            return
        self._file(path)

    def _file(self, path: Path) -> None:
        body = path.read_bytes()
        self._send(200, body, _TYPES.get(path.suffix, "application/octet-stream"))

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            return

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[annotate-web] {self.address_string()} {fmt % args}", flush=True)


def stream_file(handle: BinaryIO, write: Callable[[bytes], Any], start: int, length: int) -> None:
    """Send ``length`` bytes from ``start``. A closed browser stops the copy."""
    handle.seek(start)
    remaining = length
    while remaining:
        chunk = handle.read(min(1024 * 1024, remaining))
        if not chunk:
            break
        try:
            write(chunk)
        except ConnectionError:
            return
        remaining -= len(chunk)


def _process_state(pid: int) -> str | None:
    status = Path(f"/proc/{pid}/status")
    try:
        text = status.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("State:"):
            return line.split()[1]
    return None


def _pipeline_pid() -> int | None:
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        args = [part.decode(errors="replace") for part in raw if part]
        if not args or "python" not in Path(args[0]).name:
            continue
        if any(Path(arg).name == "run_pipeline.py" for arg in args[1:]):
            return int(entry.name)
    return None


def _feature_spec(payload: dict[str, Any]) -> str:
    raw = payload.get("features", "all")
    if raw is None or raw == "all":
        return "all"
    if isinstance(raw, list):
        return ",".join(str(item) for item in raw)
    return str(raw)


def _require_dataset() -> Path:
    if _DATASET is None:
        raise ValueError("open a dataset first")
    return _DATASET


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"LeRobot annotation review at http://{HOST}:{PORT}", flush=True)
    print(f"Guide at http://{HOST}:{PORT}/guide", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
