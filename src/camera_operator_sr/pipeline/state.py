"""Atomic state/event persistence for pipeline stages."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STATUSES = {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED", "BLOCKED"}


def now() -> str: return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class PipelineState:
    def __init__(self, path: Path, name: str, *, resume: bool):
        self.path = path
        self.value = json.loads(path.read_text()) if resume and path.exists() else {"schema_version": 1, "pipeline_name": name, "status": "PENDING", "stages": {}}

    def transition_stage(self, stage_id: str, status: str, *, payload: dict | None = None) -> None:
        """Replace stage-run metadata instead of leaking data from an older run."""
        if status not in STATUSES: raise ValueError(f"invalid pipeline stage status: {status}")
        payload = dict(payload or {})
        previous = self.value["stages"].get(stage_id, {})
        # Fingerprints survive only when explicitly supplied.  In particular an
        # old error/reason/command/output must never describe a new attempt.
        current = {"status": status, **payload}
        if status != "PENDING":
            current["started_at_utc"] = now() if status == "RUNNING" else previous.get("started_at_utc", now())
        if status in {"SUCCEEDED", "FAILED", "SKIPPED", "BLOCKED"}:
            current["finished_at_utc"] = now()
        self.value["stages"][stage_id] = current
        self.value["status"] = "FAILED" if status == "FAILED" else self.value["status"]
        write_json_atomic(self.path, self.value)

    def update(self, stage_id: str, status: str, **metadata) -> None:
        """Backward-compatible spelling for callers and older integrations."""
        self.transition_stage(stage_id, status, payload=metadata)

    def finish(self, status: str) -> None:
        self.value["status"] = status; self.value["finished_at_utc"] = now(); write_json_atomic(self.path, self.value)
