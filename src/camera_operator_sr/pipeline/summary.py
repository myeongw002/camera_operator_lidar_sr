"""Pipeline summary, including PR3's camera-guided comparison contract."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .state import write_json_atomic


def _number(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _artifact_values(artifacts: dict) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else
        {name: str(path) for name, path in value.items()} if isinstance(value, dict) else value
        for key, value in artifacts.items()
    }


def _write_comparison(writer, comparison: str, baseline: dict, candidate: dict) -> None:
    for region in sorted(set(baseline) & set(candidate)):
        if not isinstance(baseline[region], dict) or not isinstance(candidate[region], dict):
            continue
        for metric in sorted(set(baseline[region]) & set(candidate[region])):
            left, right = _number(baseline[region][metric]), _number(candidate[region][metric])
            if left is not None and right is not None:
                writer.writerow([comparison, metric, region, left, right, right - left, (right - left) / left if left else None, metric.endswith(("mae", "rmse"))])


def build_summary(context, state: dict) -> None:
    root = context.summary_root
    root.mkdir(parents=True, exist_ok=True)
    config, evaluation = context.config, context.config["evaluation"]
    guided = bool(config["teachers"].get("correct", {}).get("enabled") and config["teachers"]["correct"].get("model_type") == "relation_guided")
    distillation_enabled = bool(config["distillation"].get("enabled", True))
    controls_enabled = any(value.get("enabled", False) for name, value in config["teachers"].items() if name != "correct")

    metrics: dict[str, dict] = {}
    names = ["student_baseline"] + ([config["distillation"]["experiment_name"]] if distillation_enabled else [])
    for name in names:
        path = context.evaluations_root / name / "summary.json"
        if path.exists():
            metrics[name] = json.loads(path.read_text())

    guided_summary = None
    guided_path = context.evaluations_root / "teachers" / "summary.json"
    if guided and guided_path.exists():
        guided_summary = json.loads(guided_path.read_text())
        metrics["relation_guided"] = guided_summary

    checks = {
        "teacher_correct_better_than_baseline": False,
        "teacher_correct_better_than_teacher_none": False,
        "distillation_better_than_student": False,
        "range_kd_active": False,
        "return_kd_active": False,
    }
    if guided:
        checks.update(
            guided_better_than_l0_on_camera_guidance_mae=False,
            guided_better_than_l0_on_camera_guidance_rmse=False,
            guided_camera_guidance_count_positive=False,
        )
    warnings = list(context.warnings)

    teacher_csv = context.evaluations_root / "teachers" / "teacher_comparison.csv"
    if not guided and teacher_csv.exists():
        with teacher_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        def teacher_mae(model: str):
            values = [_number(row.get("range_mae")) for row in rows if row.get("model") == model and row.get("region") == "full"]
            return next((value for value in values if value is not None), None)

        baseline, correct, none = teacher_mae("baseline"), teacher_mae("teacher_correct"), teacher_mae("teacher_none")
        checks["teacher_correct_better_than_baseline"] = baseline is not None and correct is not None and correct < baseline
        checks["teacher_correct_better_than_teacher_none"] = none is not None and correct is not None and correct < none

    if guided_summary:
        group = guided_summary.get("camera_guidance_valid", {})
        l0_mae, guided_mae = _number(group.get("l0_range_mae")), _number(group.get("guided_range_mae"))
        l0_rmse, guided_rmse = _number(group.get("l0_range_rmse")), _number(group.get("guided_range_rmse"))
        count = _number(group.get("camera_guidance_count"))
        checks["guided_better_than_l0_on_camera_guidance_mae"] = l0_mae is not None and guided_mae is not None and guided_mae < l0_mae
        checks["guided_better_than_l0_on_camera_guidance_rmse"] = l0_rmse is not None and guided_rmse is not None and guided_rmse < l0_rmse
        checks["guided_camera_guidance_count_positive"] = count is not None and count > 0

    def full_mae(value: dict):
        return _number(value.get("full", {}).get("range_mae"))

    if distillation_enabled:
        baseline = full_mae(metrics.get("student_baseline", {}))
        distilled = full_mae(metrics.get(config["distillation"]["experiment_name"], {}))
        checks["distillation_better_than_student"] = baseline is not None and distilled is not None and distilled < baseline
        metric_path = context.experiments_root / config["distillation"]["experiment_name"] / f"seed_{config['pipeline']['seed']}" / "metrics.jsonl"
        if metric_path.exists():
            rows = [json.loads(line) for line in metric_path.read_text().splitlines() if line.strip()]
            if rows:
                latest = rows[-1]
                for label, prefix in (("range_kd_active", "range"), ("return_kd_active", "return")):
                    ratio, count = latest.get(f"{prefix}_kd_active_ratio"), latest.get(f"{prefix}_kd_eligible_count")
                    checks[label] = bool(_number(ratio) is not None and _number(count) is not None and count > 0 and ratio > 0)

    if guided:
        if not checks["guided_better_than_l0_on_camera_guidance_mae"]:
            warnings.append("relation_guided did not outperform frozen L0 on camera_guidance_valid MAE")
        if not checks["guided_better_than_l0_on_camera_guidance_rmse"]:
            warnings.append("relation_guided did not outperform frozen L0 on camera_guidance_valid RMSE")
        if not checks["guided_camera_guidance_count_positive"]:
            warnings.append("relation_guided evaluation had zero camera-guidance-valid queries")
    elif config["teachers"].get("correct", {}).get("enabled") and not checks["teacher_correct_better_than_baseline"]:
        warnings.append("teacher_correct did not outperform baseline")
    if controls_enabled and not checks["teacher_correct_better_than_teacher_none"]:
        warnings.append("teacher_correct did not outperform teacher_none")
    if distillation_enabled:
        for key, message in (("distillation_better_than_student", "distillation did not outperform student baseline"), ("range_kd_active", "range KD was never activated"), ("return_kd_active", "return KD was never activated")):
            if not checks[key]:
                warnings.append(message)

    artifacts = _artifact_values(context.artifacts)
    if guided:
        artifacts["guided_evaluation"] = str(guided_path)
    write_json_atomic(root / "artifact_index.json", artifacts)
    write_json_atomic(root / "pipeline_summary.json", {"pipeline_name": config["pipeline"]["name"], "status": state["status"], "seed": config["pipeline"]["seed"], "device": context.device, "dataset": {"train_frames": len(context.train_entries), "validation_frames": len(context.validation_entries), "test_frames": len(context.test_entries)}, "artifacts": artifacts, "metrics": metrics, "scientific_checks": checks, "warnings": warnings})

    experiments = [("student_baseline", context.artifacts.get("student_best_checkpoint"))]
    if guided:
        experiments.append(("relation_guided", context.artifacts.get("teacher_checkpoints", {}).get("correct")))
    if distillation_enabled:
        experiments.append(("distillation", context.artifacts.get("distillation_best_checkpoint")))
    with (root / "experiment_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["experiment", "checkpoint"])
        for name, checkpoint in experiments:
            writer.writerow([name, checkpoint or ""])

    with (root / "metric_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "metric", "region", "baseline", "candidate", "delta", "relative_delta", "lower_is_better"])
        if distillation_enabled:
            _write_comparison(writer, "student_vs_distillation", metrics.get("student_baseline", {}), metrics.get(config["distillation"]["experiment_name"], {}))
        if guided_summary:
            group = guided_summary.get("camera_guidance_valid", {})
            for metric in ("range_mae", "range_rmse"):
                baseline, candidate = _number(group.get(f"l0_{metric}")), _number(group.get(f"guided_{metric}"))
                if baseline is not None and candidate is not None:
                    writer.writerow(["l0_vs_guided", metric, "camera_guidance_valid", baseline, candidate, candidate - baseline, (candidate - baseline) / baseline if baseline else None, True])
