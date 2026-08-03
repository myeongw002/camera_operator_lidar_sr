"""Stage identities and dependency graph."""
from __future__ import annotations

STAGES = ("P00_preflight", "P01_dataset_validation", "P02_prepare_range_images", "P03_precompute_depth", "P04_create_splits", "P05_train_student", "P06_train_teacher_correct", "P07_train_teacher_controls", "P08_evaluate_teachers", "P09_train_distillation", "P10_evaluate_sr", "P11_inference", "P12_summary")
DEPENDENCIES = {
    "P00_preflight": (), "P01_dataset_validation": ("P00_preflight",),
    "P02_prepare_range_images": ("P01_dataset_validation",), "P03_precompute_depth": ("P01_dataset_validation",),
    "P04_create_splits": ("P02_prepare_range_images",), "P05_train_student": ("P04_create_splits",),
    "P06_train_teacher_correct": ("P03_precompute_depth", "P05_train_student"),
    "P07_train_teacher_controls": ("P03_precompute_depth", "P05_train_student"),
    "P08_evaluate_teachers": ("P06_train_teacher_correct", "P07_train_teacher_controls"),
    "P09_train_distillation": ("P05_train_student", "P06_train_teacher_correct"),
    "P10_evaluate_sr": ("P05_train_student", "P09_train_distillation"), "P11_inference": ("P09_train_distillation",),
    "P12_summary": ("P08_evaluate_teachers", "P10_evaluate_sr", "P11_inference"),
}


def topological_order() -> tuple[str, ...]: return STAGES


def downstream(stage_id: str) -> set[str]:
    result, pending = set(), [stage_id]
    while pending:
        current = pending.pop()
        for stage, dependencies in DEPENDENCIES.items():
            if current in dependencies and stage not in result:
                result.add(stage); pending.append(stage)
    return result
