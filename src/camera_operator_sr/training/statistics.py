"""Numerically explicit epoch aggregation for distillation diagnostics."""

DISTILLATION_STATISTIC_KEYS = (
    "range_advantage_sum", "range_advantage_count", "return_advantage_sum", "return_advantage_count",
    "range_kd_active_count", "range_kd_eligible_count", "return_kd_active_count", "return_kd_eligible_count",
)


def empty_distillation_statistics() -> dict[str, float]:
    return {key: 0.0 for key in DISTILLATION_STATISTIC_KEYS}


def add_distillation_statistics(totals: dict[str, float], values: dict) -> None:
    for key in DISTILLATION_STATISTIC_KEYS:
        totals[key] += float(values[key].detach())


def finalize_distillation_statistics(totals: dict[str, float]) -> dict:
    """Return query-weighted means; no eligible query is represented as null."""
    def divide(numerator: str, denominator: str):
        return totals[numerator] / totals[denominator] if totals[denominator] else None
    return dict(totals,
        mean_range_advantage=divide("range_advantage_sum", "range_advantage_count"),
        mean_return_advantage=divide("return_advantage_sum", "return_advantage_count"),
        range_kd_active_ratio=divide("range_kd_active_count", "range_kd_eligible_count"),
        return_kd_active_ratio=divide("return_kd_active_count", "return_kd_eligible_count"),
    )
