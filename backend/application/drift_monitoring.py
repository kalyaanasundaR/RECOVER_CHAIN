"""Phase 14 — four-tier drift classification, built ON TOP of Phase 11's
application.drift_readiness (reused, not duplicated: same summarize_distribution/
compare_distributions, same standardized-mean-shift / total-variation-distance
scores). Phase 11 only exposed a binary drifted/not-drifted flag; this module
adds the NO_DRIFT / LOW_DRIFT / MODERATE_DRIFT / HIGH_DRIFT tiers Phase 14 asks
for, as multiples of Phase 11's own threshold - an OPERATIONAL heuristic tiering,
explicitly documented as such, not a new statistically-derived cutoff.

Also separates three distinct concepts the brief insists are NOT the same
thing:
  - FEATURE DRIFT: the input distribution changed (this module).
  - MODEL PERFORMANCE DRIFT: the model's OWN accuracy/calibration changed over
    time (application.model_monitoring.temporal_monitoring - compare metric
    blocks across periods).
  - BUSINESS OUTCOME DRIFT: the underlying recovery rate itself changed,
    independent of whether the model predicted it well (also read off
    temporal_monitoring's recovery_rate field, but interpreted separately -
    see classify_performance_vs_business_drift below).
A feature-drift HIGH_DRIFT finding does NOT by itself imply the model became
inaccurate; the two are reported side by side, never merged into one number.
"""

from __future__ import annotations

from application.drift_readiness import (
    CATEGORICAL_DRIFT_THRESHOLD,
    NUMERIC_DRIFT_THRESHOLD,
    compare_distributions,
    summarize_distribution,
)

# Multiples of Phase 11's existing threshold. Documented operational tiering,
# not a statistically fitted boundary (no production traffic exists yet to
# fit one against).
_TIER_MULTIPLIERS = {
    "LOW_DRIFT": 0.25,
    "MODERATE_DRIFT": 1.0,
    "HIGH_DRIFT": 2.0,
}


def _tier(score: float | None, threshold: float) -> str:
    if score is None:
        return "UNKNOWN"
    if score >= threshold * _TIER_MULTIPLIERS["HIGH_DRIFT"]:
        return "HIGH_DRIFT"
    if score >= threshold * _TIER_MULTIPLIERS["MODERATE_DRIFT"]:
        return "MODERATE_DRIFT"
    if score >= threshold * _TIER_MULTIPLIERS["LOW_DRIFT"]:
        return "LOW_DRIFT"
    return "NO_DRIFT"


def classify_feature_drift(training_distribution: dict, candidate_distribution: dict) -> dict:
    """Wraps drift_readiness.compare_distributions with the four-tier
    classification, per column and overall (the column with the highest tier
    determines the overall tier - a single badly-drifted feature is worth
    surfacing even if the rest are fine)."""
    comparison = compare_distributions(training_distribution, candidate_distribution)
    tier_rank = {"UNKNOWN": -1, "NO_DRIFT": 0, "LOW_DRIFT": 1, "MODERATE_DRIFT": 2, "HIGH_DRIFT": 3}

    per_column = {}
    overall = "NO_DRIFT"
    for col, result in comparison["columns"].items():
        b = training_distribution.get(col, {})
        threshold = (
            CATEGORICAL_DRIFT_THRESHOLD
            if b.get("type") == "categorical"
            else NUMERIC_DRIFT_THRESHOLD
        )
        tier = _tier(result.get("drift_score"), threshold)
        per_column[col] = {**result, "tier": tier}
        if tier_rank[tier] > tier_rank[overall]:
            overall = tier

    return {
        "overall_tier": overall,
        "methodology": (
            f"standardized mean shift for numeric columns (thresholds: LOW>="
            f"{NUMERIC_DRIFT_THRESHOLD * 0.25}, MODERATE>={NUMERIC_DRIFT_THRESHOLD}, "
            f"HIGH>={NUMERIC_DRIFT_THRESHOLD * 2}); total variation distance for "
            f"categorical columns (thresholds: LOW>={CATEGORICAL_DRIFT_THRESHOLD * 0.25}, "
            f"MODERATE>={CATEGORICAL_DRIFT_THRESHOLD}, HIGH>={CATEGORICAL_DRIFT_THRESHOLD * 2}) - "
            "operational heuristics built on Phase 11's thresholds, not a fitted "
            "statistical boundary"
        ),
        "columns": per_column,
    }


def classify_performance_vs_business_drift(period_metrics: dict) -> dict:
    """Takes application.model_monitoring.temporal_monitoring's output
    ({period: MetricBlock}) and reports the two trends SEPARATELY:
    mean_absolute_probability_error trend (performance) vs recovery_rate trend
    (business outcome). Only computed where consecutive periods both have
    COMPUTED status; otherwise reported as INSUFFICIENT_EVIDENCE."""
    periods = sorted(k for k, v in period_metrics.items() if v.status == "COMPUTED")
    if len(periods) < 2:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": f"{len(periods)} period(s) with enough data to compare - need >= 2",
        }

    first, last = periods[0], periods[-1]
    first_m, last_m = period_metrics[first], period_metrics[last]

    performance_delta = None
    if (
        first_m.mean_absolute_probability_error is not None
        and last_m.mean_absolute_probability_error is not None
    ):
        performance_delta = round(
            last_m.mean_absolute_probability_error - first_m.mean_absolute_probability_error, 4
        )
    business_delta = None
    if first_m.recovery_rate is not None and last_m.recovery_rate is not None:
        business_delta = round(last_m.recovery_rate - first_m.recovery_rate, 4)

    return {
        "status": "COMPUTED",
        "first_period": first,
        "last_period": last,
        "performance_drift": {
            "metric": "mean_absolute_probability_error",
            "delta": performance_delta,
            "interpretation": "positive = predictions got LESS accurate over time",
        },
        "business_outcome_drift": {
            "metric": "recovery_rate",
            "delta": business_delta,
            "interpretation": "positive = a higher share of cases recovered over time",
        },
        "note": (
            "these are reported separately on purpose: a feature-distribution "
            "change, a model-accuracy change, and a real-world recovery-rate "
            "change are three different things and one does not imply another"
        ),
    }


def summarize_distribution_for_drift(df, columns):
    """Re-exported for callers that only need drift_readiness's summary, so
    this module can be the one-stop import for Phase 14 drift work without
    forcing every caller to also import application.drift_readiness."""
    return summarize_distribution(df, columns)
