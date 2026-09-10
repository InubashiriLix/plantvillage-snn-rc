#!/usr/bin/env python3
"""Record health/progress for the best-model campaign without touching training."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "artifacts/compact_pipeline"


def command(*args: str) -> tuple[int, str]:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    output = (result.stdout or result.stderr).strip()
    return result.returncode, output


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    state_path = CAMPAIGN_ROOT / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    _, service_state = command(
        "systemctl", "--user", "is-active", "plantvillage-snn-best.service"
    )
    _, gpu = command(
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader",
    )
    events_path = CAMPAIGN_ROOT / "events.jsonl"
    last_event = None
    if events_path.is_file():
        lines = events_path.read_text().splitlines()
        if lines:
            last_event = json.loads(lines[-1])
    tasks = state.get("tasks", {})
    latest = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "pipeline_status": state.get("status", "running" if tasks or service_state == "active" else "not_started"),
        "service_state": service_state,
        "completed_tasks": len(tasks),
        "compute_hours": round(float(state.get("compute_seconds", 0)) / 3600, 4),
        "last_event": last_event,
        "gpu": gpu,
    }
    atomic_json(CAMPAIGN_ROOT / "latest_status.json", latest)
    with (CAMPAIGN_ROOT / "monitor.jsonl").open("a") as stream:
        stream.write(json.dumps(latest, sort_keys=True) + "\n")
    print(json.dumps(latest, ensure_ascii=False), flush=True)
    if state.get("status") == "complete":
        command("systemctl", "--user", "stop", "plantvillage-snn-monitor.timer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
