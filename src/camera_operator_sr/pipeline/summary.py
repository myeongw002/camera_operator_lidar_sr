"""Pipeline summary and artifact index generation."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .state import write_json_atomic


def build_summary(context, state: dict) -> None:
    root = context.summary_root; root.mkdir(parents=True, exist_ok=True)
    artifacts = {key: str(value) if isinstance(value, Path) else {name: str(path) for name, path in value.items()} if isinstance(value, dict) else value for key, value in context.artifacts.items()}
    metrics = {}
    for name in ("student_baseline", context.config["distillation"]["experiment_name"]):
        summary = context.evaluations_root/name/"summary.json"
        if summary.exists():
            values = json.loads(summary.read_text()); metrics[name] = values
    checks = {"teacher_correct_better_than_baseline": False, "teacher_correct_better_than_teacher_none": False, "distillation_better_than_student": False, "range_kd_active": False, "return_kd_active": False}
    warnings = list(context.warnings)
    teacher_csv = context.evaluations_root / "teachers" / "teacher_comparison.csv"
    if teacher_csv.exists():
        with teacher_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        def mae(model: str):
            values = [float(row["range_mae"]) for row in rows if row.get("model") == model and row.get("region") == "full" and row.get("range_mae") not in {"", "nan", "NaN"}]
            return values[0] if values else None
        baseline, correct, none = mae("baseline"), mae("teacher_correct"), mae("teacher_none")
        checks["teacher_correct_better_than_baseline"] = baseline is not None and correct is not None and correct < baseline
        checks["teacher_correct_better_than_teacher_none"] = none is not None and correct is not None and correct < none
    def full_mae(value: dict):
        row = value.get("full", {})
        result = row.get("range_mae")
        return float(result) if isinstance(result, (int, float)) and math.isfinite(result) else None
    baseline, distilled = full_mae(metrics.get("student_baseline", {})), full_mae(metrics.get(context.config["distillation"]["experiment_name"], {}))
    checks["distillation_better_than_student"] = baseline is not None and distilled is not None and distilled < baseline
    metric_path = context.experiments_root / context.config["distillation"]["experiment_name"] / f"seed_{context.config['pipeline']['seed']}" / "metrics.jsonl"
    if metric_path.exists():
        rows = [json.loads(line) for line in metric_path.read_text().splitlines() if line.strip()]
        if rows:
            latest = rows[-1]
            for label, prefix in (("range_kd_active", "range"), ("return_kd_active", "return")):
                ratio, count = latest.get(f"{prefix}_kd_active_ratio"), latest.get(f"{prefix}_kd_eligible_count")
                checks[label] = bool(isinstance(ratio, (int, float)) and isinstance(count, (int, float)) and count > 0 and ratio > 0)
    messages = (("teacher_correct_better_than_baseline", "teacher_correct did not outperform baseline"), ("teacher_correct_better_than_teacher_none", "teacher_correct did not outperform teacher_none"), ("distillation_better_than_student", "distillation did not outperform student baseline"), ("range_kd_active", "range KD was never activated"), ("return_kd_active", "return KD was never activated"))
    for key, message in messages:
        if not checks[key]: warnings.append(message)
    write_json_atomic(root/"artifact_index.json", artifacts)
    write_json_atomic(root/"pipeline_summary.json", {"pipeline_name": context.config["pipeline"]["name"], "status": state["status"], "seed": context.config["pipeline"]["seed"], "device": context.device, "dataset": {"train_frames": len(context.train_entries), "validation_frames": len(context.validation_entries), "test_frames": len(context.test_entries)}, "artifacts": artifacts, "metrics": metrics, "scientific_checks": checks, "warnings": warnings})
    with (root/"experiment_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["experiment", "checkpoint"])
        for name, checkpoint in (("student_baseline", context.artifacts.get("student_best_checkpoint")), ("distillation", context.artifacts.get("distillation_best_checkpoint"))): writer.writerow([name, checkpoint or ""])
    with (root/"metric_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["metric", "region", "baseline", "distillation", "delta", "relative_delta", "lower_is_better"])
        base, dist = metrics.get("student_baseline", {}), metrics.get(context.config["distillation"]["experiment_name"], {})
        for region in sorted(set(base) & set(dist)):
            if not isinstance(base[region], dict) or not isinstance(dist[region], dict): continue
            for key in sorted(set(base[region]) & set(dist[region])):
                left, right = base[region][key], dist[region][key]
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    writer.writerow([key, region, left, right, right - left, (right - left) / left if left else None, key.endswith("mae") or key.endswith("rmse")])
