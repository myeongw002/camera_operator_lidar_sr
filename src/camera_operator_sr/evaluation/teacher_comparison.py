def select_primary_region(rows: list[dict], required_models: tuple[str, ...] = ("baseline", "teacher_correct")) -> str:
    for region in ("gt_camera_visible", "camera_boundary", "camera_interior", "camera_frustum", "front_azimuth", "full"):
        matches = {row["model"]: row for row in rows if row.get("region") == region}
        if all(model in matches and matches[model].get("range_count", 0) > 0 and matches[model].get("range_mae") is not None for model in required_models): return region
    raise RuntimeError("No non-empty common region is available for teacher comparison.")


def build_superiority(rows: list[dict]) -> dict:
    """Create transparent teacher-vs-control MAE deltas from comparison rows."""
    primary_region = select_primary_region(rows)
    by_name = {row["model"]: row for row in rows if row.get("region") == primary_region}
    correct = by_name.get("teacher_correct")
    result = {"primary_region": primary_region, "primary_region_range_count": min(by_name["baseline"]["range_count"], by_name["teacher_correct"]["range_count"])}
    if correct:
        for control, label in (("baseline", "baseline"), ("teacher_none", "none"), ("teacher_frame_shuffled", "frame_shuffled"), ("teacher_spatial_shuffled", "spatial_shuffled")):
            if control in by_name:
                delta = correct["range_mae"] - by_name[control]["range_mae"]
                result["primary_region"] = primary_region
                result[f"delta_mae_correct_minus_{label}"] = delta
                result[f"mae_improvement_over_{label}"] = -delta
                result[f"correct_better_than_{label}"] = delta < 0
    return result
