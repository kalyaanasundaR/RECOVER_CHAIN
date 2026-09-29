"""Phase 14 — prediction-vs-reality metrics, temporal/domain/value-band
monitoring, and calibration tracking, all computed from
application.prediction_outcome_linkage's real, eligible records.

MIN_SAMPLE_SIZE (30) and MIN_PER_CLASS (2) are documented OPERATIONAL
heuristics, not statistically derived thresholds - stated explicitly here
rather than presented as scientifically justified cutoffs (Section 5's own
instruction: "if thresholds cannot be statistically justified, document them
as operational heuristics"). Below them, a metric block reports
INSUFFICIENT_EVIDENCE instead of a number that would misleadingly imply
confidence the sample doesn't support.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from application.prediction_outcome_linkage import PredictionOutcomeRecord

MIN_SAMPLE_SIZE = 30
MIN_PER_CLASS = 2

VALUE_BANDS = [
    ("low", 0.0, 500.0),
    ("medium", 500.0, 5000.0),
    ("high", 5000.0, float("inf")),
]


@dataclass
class MetricBlock:
    status: str  # "COMPUTED" | "INSUFFICIENT_EVIDENCE"
    n: int
    n_recovered: int
    n_not_recovered: int
    reason: str | None = None
    roc_auc: float | None = None
    pr_auc: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    brier_score: float | None = None
    log_loss_: float | None = None
    mean_absolute_probability_error: float | None = None
    recovery_rate: float | None = None
    mean_predicted_probability: float | None = None
    total_predicted_erv: float | None = None
    total_actual_recovered: float | None = None
    mean_absolute_erv_error: float | None = None
    provenance_breakdown: dict = field(default_factory=dict)


def _provenance_breakdown(records: list[PredictionOutcomeRecord]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.provenance] += 1
    return dict(counts)


def compute_metrics(records: list[PredictionOutcomeRecord]) -> MetricBlock:
    """Classification/probability/financial metrics on one already-filtered
    record set. Caller decides the filter (all, one period, one domain, one
    value band) - this function only ever computes on what it's given."""
    n = len(records)
    n_recovered = sum(1 for r in records if r.recovered)
    n_not_recovered = n - n_recovered
    prov = _provenance_breakdown(records)

    if n < MIN_SAMPLE_SIZE:
        return MetricBlock(
            status="INSUFFICIENT_EVIDENCE",
            n=n,
            n_recovered=n_recovered,
            n_not_recovered=n_not_recovered,
            reason=f"n={n} < operational minimum {MIN_SAMPLE_SIZE}",
            provenance_breakdown=prov,
        )

    mean_abs_prob_err = sum(r.absolute_probability_error for r in records) / n
    recovery_rate = n_recovered / n
    mean_prob = sum(r.predicted_recovery_probability for r in records) / n
    erv_values = [r.predicted_erv for r in records if r.predicted_erv is not None]
    total_predicted_erv = sum(erv_values) if erv_values else None
    total_actual = sum(r.actual_recovered_amount for r in records)
    abs_erv_errs = [r.absolute_erv_error for r in records if r.absolute_erv_error is not None]
    mean_abs_erv_err = sum(abs_erv_errs) / len(abs_erv_errs) if abs_erv_errs else None

    block = MetricBlock(
        status="COMPUTED",
        n=n,
        n_recovered=n_recovered,
        n_not_recovered=n_not_recovered,
        mean_absolute_probability_error=round(mean_abs_prob_err, 4),
        recovery_rate=round(recovery_rate, 4),
        mean_predicted_probability=round(mean_prob, 4),
        total_predicted_erv=round(total_predicted_erv, 2) if total_predicted_erv else None,
        total_actual_recovered=round(total_actual, 2),
        mean_absolute_erv_error=round(mean_abs_erv_err, 2)
        if mean_abs_erv_err is not None
        else None,
        provenance_breakdown=prov,
    )

    if n_recovered < MIN_PER_CLASS or n_not_recovered < MIN_PER_CLASS:
        block.reason = (
            f"classification metrics (ROC-AUC/PR-AUC/precision/recall/F1/log-loss) require "
            f"both classes with >= {MIN_PER_CLASS} examples each; got "
            f"{n_recovered} recovered / {n_not_recovered} not-recovered - reporting the "
            "metrics that don't need class diversity (Brier, calibration error, financial "
            "error) only"
        )
        block.brier_score = round(
            sum(
                (r.predicted_recovery_probability - (1.0 if r.recovered else 0.0)) ** 2
                for r in records
            )
            / n,
            4,
        )
        return block

    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.array([1 if r.recovered else 0 for r in records])
    y_prob = np.array([r.predicted_recovery_probability for r in records])
    y_pred = (y_prob >= 0.5).astype(int)

    block.roc_auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    block.pr_auc = round(float(average_precision_score(y_true, y_prob)), 4)
    block.precision = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    block.recall = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    block.f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    block.brier_score = round(float(brier_score_loss(y_true, y_prob)), 4)
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    block.log_loss_ = round(float(log_loss(y_true, clipped)), 4)
    return block


def temporal_monitoring(
    records: list[PredictionOutcomeRecord], *, period: str = "month"
) -> dict[str, MetricBlock]:
    """period: 'week' | 'month' | 'quarter'. Buckets by decision_timestamp
    (when the prediction was made, not when the outcome arrived - monitoring
    asks "how did predictions made in period X turn out", not "how many
    outcomes landed in X")."""
    buckets: dict[str, list[PredictionOutcomeRecord]] = defaultdict(list)
    for r in records:
        if not r.decision_timestamp:
            continue
        dt = datetime.fromisoformat(r.decision_timestamp)
        if period == "week":
            key = f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"
        elif period == "quarter":
            key = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        else:
            key = f"{dt.year}-{dt.month:02d}"
        buckets[key].append(r)
    return {key: compute_metrics(recs) for key, recs in sorted(buckets.items())}


def domain_monitoring(records: list[PredictionOutcomeRecord]) -> dict[str, MetricBlock]:
    buckets: dict[str, list[PredictionOutcomeRecord]] = defaultdict(list)
    for r in records:
        buckets[r.risk_category].append(r)
    return {domain: compute_metrics(recs) for domain, recs in buckets.items()}


def value_band_monitoring(records: list[PredictionOutcomeRecord]) -> dict[str, MetricBlock]:
    buckets: dict[str, list[PredictionOutcomeRecord]] = {name: [] for name, _, _ in VALUE_BANDS}
    for r in records:
        for name, lo, hi in VALUE_BANDS:
            if lo <= r.amount_at_risk < hi:
                buckets[name].append(r)
                break
    return {band: compute_metrics(recs) for band, recs in buckets.items()}


def calibration_bins(records: list[PredictionOutcomeRecord]) -> dict[str, dict]:
    """Predicted-vs-observed recovery frequency per probability decile. Only a
    bin with >= MIN_PER_CLASS-independent-of-class-balance samples (using the
    plain sample-size floor here, since a calibration bin doesn't need class
    balance the way ROC-AUC does) is reported with numbers; smaller bins are
    marked INSUFFICIENT_EVIDENCE rather than shown with a misleading 100%/0%
    from 1-2 points."""
    MIN_BIN_SIZE = 5  # operational heuristic - a 1-4 point bin is not a rate
    buckets: dict[str, list[PredictionOutcomeRecord]] = defaultdict(list)
    for r in records:
        buckets[r.calibration_bucket].append(r)

    result = {}
    for bucket, recs in sorted(buckets.items()):
        n = len(recs)
        if n < MIN_BIN_SIZE:
            result[bucket] = {"status": "INSUFFICIENT_EVIDENCE", "n": n}
            continue
        predicted_mean = sum(r.predicted_recovery_probability for r in recs) / n
        observed_rate = sum(1 for r in recs if r.recovered) / n
        result[bucket] = {
            "status": "COMPUTED",
            "n": n,
            "predicted_mean_probability": round(predicted_mean, 4),
            "observed_recovery_rate": round(observed_rate, 4),
            "calibration_gap": round(abs(predicted_mean - observed_rate), 4),
        }
    return result


def anomaly_monitoring(records: list[PredictionOutcomeRecord]) -> dict:
    """Aggregate anomaly signal stats - explicitly NOT an interpretation of
    what anomaly means (never "fraud", "bad customer", or "no recovery" -
    Section 16). Just counts and a rate, plus whether recovery outcomes
    differ between anomalous and normal groups (an OBSERVATION, not a causal
    or fraud claim)."""
    n = len(records)
    if n == 0:
        return {"status": "INSUFFICIENT_EVIDENCE", "n": 0}

    levels: dict[str, int] = defaultdict(int)
    for r in records:
        levels[r.anomaly_level or "UNKNOWN"] += 1

    anomalous = [r for r in records if r.anomaly_level in ("ANOMALOUS", "UNUSUAL")]
    normal = [r for r in records if r.anomaly_level == "NORMAL"]

    return {
        "status": "COMPUTED",
        "n": n,
        "level_counts": dict(levels),
        "anomaly_rate": round(len(anomalous) / n, 4),
        "recovery_rate_anomalous": (
            round(sum(1 for r in anomalous if r.recovered) / len(anomalous), 4)
            if len(anomalous) >= MIN_PER_CLASS
            else None
        ),
        "recovery_rate_normal": (
            round(sum(1 for r in normal if r.recovered) / len(normal), 4)
            if len(normal) >= MIN_PER_CLASS
            else None
        ),
        "note": (
            "observed association only - anomaly is a decision signal, never "
            "an automatic interpretation of fraud, customer quality, or "
            "recovery likelihood (Section 16)"
        ),
    }
