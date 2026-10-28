"""Zero-extra-dependency local UI for the layered hardware-RC pipeline."""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from .layers import (
    HttpHardwareBackend,
    LayerUnavailable,
    LeafImageProcessor,
    ReadoutDecisionLayer,
    RecordedHardwareBackend,
    RecordedHardwareDataset,
    response_from_nested,
)
from .layers.contracts import DRIVE_CHANNELS

ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = Path(__file__).resolve().parent / "webui"


def _b64_image_preview(image_bytes: bytes, size: int = 320) -> str:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _default_model() -> Path | None:
    candidates = (
        ROOT / "artifacts_hardware300_v2/run/model.joblib",
        ROOT / "artifacts_hardware300_v2/seed_run/model.joblib",
        ROOT / "artifacts_hardware300/model.joblib",
    )
    return next((path for path in candidates if path.is_file()), None)


class Application:
    def __init__(
        self,
        *,
        model_path: Path | None,
        responses_path: Path | None,
        inputs_path: Path | None,
        hardware_url: str | None,
    ):
        self.processor = LeafImageProcessor()
        self.decision = ReadoutDecisionLayer(model_path)
        self.dataset: RecordedHardwareDataset | None = None
        self.dataset_error: str | None = None
        if responses_path and inputs_path and responses_path.is_file() and inputs_path.is_file():
            try:
                self.dataset = RecordedHardwareDataset(responses_path, inputs_path)
            except Exception as error:
                self.dataset_error = str(error)
        self.recorded = RecordedHardwareBackend(self.dataset) if self.dataset else None
        self.http = HttpHardwareBackend(hardware_url) if hardware_url else None

    def pipeline_for(self, hardware=None):
        from .layers import HardwareRCPipeline

        backend = hardware or self.recorded
        if backend is None:
            raise LayerUnavailable("no hardware backend is configured")
        return HardwareRCPipeline(self.processor, backend, self.decision)

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "layers": [
                {"name": "processing", "label": "处理层", "ready": True,
                 "detail": "64×9 graded_q000 驱动信号"},
                {"name": "hardware", "label": "硬件响应层",
                 "ready": self.recorded is not None or self.http is not None,
                 "detail": "HTTP 实采" if self.http else ("本地实测回放" if self.recorded else "未连接")},
                {"name": "decision", "label": "决策层", "ready": self.decision.available,
                 "detail": self.decision.model_name},
            ],
            "decision": self.decision.status(),
            "recorded_dataset": (
                {
                    "ready": True,
                    "sample_count": len(self.dataset.metadata),
                    "source": self.dataset.responses_path.name,
                }
                if self.dataset
                else {"ready": False, "error": self.dataset_error or "未找到录制数据"}
            ),
            "hardware": {
                "mode": "http" if self.http else ("recorded" if self.recorded else "unavailable"),
                "endpoint": self.http.url if self.http else None,
            },
            "protocol": {
                "drive_channels": list(DRIVE_CHANNELS),
                "drive_shape": [64, 9],
                "response_shape": [9, 12, 65],
                "response_units": "A",
            },
        }


def _result_payload(result) -> dict[str, Any]:
    return result.to_dict(include_currents=True)


class Handler(BaseHTTPRequestHandler):
    app: Application

    def log_message(self, fmt: str, *args: Any) -> None:
        print(json.dumps({"http": fmt % args}, ensure_ascii=False), flush=True)

    def _send(self, status: int, value: Any, content_type: str = "application/json") -> None:
        if isinstance(value, (dict, list)):
            body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        elif isinstance(value, bytes):
            body = value
        else:
            body = str(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 32 * 1024 * 1024:
            raise ValueError("request body missing or too large")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/status":
            self._send(200, self.app.status())
            return
        if route == "/api/samples":
            if not self.app.dataset:
                self._send(503, {"error": self.app.dataset_error or "录制数据未安装"})
                return
            self._send(200, {"samples": self.app.dataset.sample_summaries()})
            return
        self._serve_static(route)

    def _serve_static(self, route: str) -> None:
        relative = "index.html" if route in ("", "/") else route.lstrip("/")
        path = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT.resolve() not in path.parents and path != STATIC_ROOT.resolve():
            self._send(404, {"error": "not found"})
            return
        if not path.is_file():
            self._send(404, {"error": "not found"})
            return
        suffix = path.suffix.lower()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
        }.get(suffix, "application/octet-stream")
        self._send(200, path.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        try:
            route = urlparse(self.path).path
            body = self._json_body()
            if route == "/api/process":
                self._process(body)
            elif route == "/api/run":
                self._run(body)
            elif route == "/api/decision":
                self._decision(body)
            elif route == "/api/decide-csv":
                self._decide_csv(body)
            else:
                self._send(404, {"error": "not found"})
        except (ValueError, KeyError, LayerUnavailable) as error:
            self._send(400, {"error": str(error)})
        except Exception as error:
            self._send(500, {"error": f"{type(error).__name__}: {error}"})

    def _process(self, body: dict[str, Any]) -> None:
        try:
            image = base64.b64decode(body["image_base64"], validate=True)
        except Exception as error:
            raise ValueError("image_base64 is not valid base64") from error
        sample_id = str(body.get("filename") or "uploaded-image").strip()
        drive = self.app.processor.process(image, sample_id=sample_id)
        self._send(200, {
            "drive": drive.to_dict(),
            "preview": _b64_image_preview(image),
            "next": "连接硬件层后调用 /api/run 的 upload 模式，或单独上传实测电流 CSV",
        })

    def _run(self, body: dict[str, Any]) -> None:
        source = body.get("source", "recorded")
        if source == "recorded":
            if not self.app.dataset or not self.app.recorded:
                raise LayerUnavailable("录制数据未安装")
            sample_index = int(body["sample_index"])
            drive = self.app.dataset.drive(sample_index)
            result = self.app.pipeline_for(None).run_drive_with(drive, self.app.recorded)
            payload = _result_payload(result)
            payload["source"] = "recorded"
            payload["sample"] = self.app.dataset.sample_summaries()[sample_index]
            self._send(200, payload)
            return
        if source == "upload":
            if not self.app.http:
                raise LayerUnavailable(
                    "未配置真实硬件接口；请用 --hardware-url 启动 UI 后再跑上传图片，"
                    "或选择录制样本演示完整链路"
                )
            try:
                image = base64.b64decode(body["image_base64"], validate=True)
            except Exception as error:
                raise ValueError("image_base64 is not valid base64") from error
            sample_id = str(body.get("filename") or "uploaded-image")
            result = self.app.pipeline_for(self.app.http).run_image(
                image, sample_id=sample_id, hardware=self.app.http
            )
            payload = _result_payload(result)
            payload["source"] = "upload"
            payload["preview"] = _b64_image_preview(image)
            self._send(200, payload)
            return
        raise ValueError("source must be 'recorded' or 'upload'")

    def _decision(self, body: dict[str, Any]) -> None:
        if "sample_index" in body:
            if not self.app.dataset:
                raise LayerUnavailable("录制数据未安装")
            response = self.app.dataset.response(int(body["sample_index"]))
        elif "currents_amperes" in body:
            from .layers import HardwareResponse

            response = HardwareResponse(
                currents=response_from_nested(body["currents_amperes"]),
                sample_id=str(body.get("sample_id") or "request-response"),
                backend="request",
                measured=True,
            )
        else:
            raise ValueError("provide sample_index or currents_amperes")
        self._send(200, {"decision": self.app.decision.decide(response).to_dict()})

    def _decide_csv(self, body: dict[str, Any]) -> None:
        text = str(body.get("csv_text") or "")
        if not text.strip():
            raise ValueError("csv_text is empty")
        from .layers.hardware import _read_response_csv

        with NamedTemporaryFile("w", suffix=".csv", encoding="utf-8", delete=False) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        try:
            currents, metadata = _read_response_csv(temporary)
        finally:
            temporary.unlink(missing_ok=True)
        decisions = self.app.decision.decide_batch(currents)
        rows = []
        for meta, decision in zip(metadata, decisions):
            rows.append({
                "sample_id": meta["sample_id"],
                "predicted_class": decision.predicted_class,
                "confidence": decision.to_dict()["probability"],
            })
        self._send(200, {"sample_count": len(rows), "predictions": rows})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hardware RC layered UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--responses", type=Path, default=None)
    parser.add_argument("--inputs", type=Path, default=None)
    parser.add_argument("--hardware-url", default=os.environ.get("PLANTVILLAGE_HARDWARE_URL"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    model = args.model
    if model is None:
        configured = os.environ.get("PLANTVILLAGE_HARDWARE_MODEL")
        model = Path(configured) if configured else _default_model()
    responses = args.responses or ROOT / "artifacts_hardware300/sources/responses.csv"
    inputs = args.inputs or ROOT / "artifacts_hardware300/sources/inputs.csv"
    app = Application(
        model_path=model,
        responses_path=responses,
        inputs_path=inputs,
        hardware_url=args.hardware_url,
    )
    Handler.app = app
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({
        "event": "ui_ready",
        "url": f"http://{args.host}:{args.port}",
        "model": str(model) if model else None,
        "recorded_samples": len(app.dataset.metadata) if app.dataset else 0,
        "hardware_mode": app.status()["hardware"]["mode"],
    }, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
