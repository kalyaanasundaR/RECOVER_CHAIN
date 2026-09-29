"""Phase 14 — closes the PREDICTION -> OUTCOME loop.

For every case that has BOTH a recorded shadow prediction AND at least one
MATCHED authoritative outcome event, builds one immutable PredictionOutcomeRecord
joining what the model said at decision time to what actually happened later.
Never overwrites case.prediction (the original record) or case.recovered_amount
(the authoritative one) - this module only READS both and produces a third,
separate, derived record for monitoring.

Eligibility (a case only becomes an eligible row here once):
  - case.prediction exists (a real shadow or baseline prediction was made)
  - at least one MATCHED OutcomeEventModel row exists for the case, at or
    after the prediction's own timestamp (never uses an outcome that predates
    the prediction it would be "explaining" - that would be using information
    the prediction couldn't have had)

A case with a prediction but no matched outcome yet is NOT eligible - it is
PENDING, not a negative example. This module never invents a "not recovered"
label from silence; RecoverChain currently has no cancellation/write-off
outcome_type wired for every domain (e.g. FAILED_PAYMENT has only
PAYMENT_SETTLED, domain/revenue_domains.py OUTCOME_TYPE_MAP), so "no
settlement observed yet" and "confirmed not recovered" are NOT the same
thing and must not be conflated into a synthetic negative label.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from infrastructure.orm import CaseModel, OutcomeEventModel


@dataclass
class PredictionOutcomeRecord:
    case_id: str
    risk_category: str
    provenance: str
    decision_timestamp: str | None
    predicted_recovery_probability: float
    predicted_erv: float | None
    amount_at_risk: float
    model_version: str
    prediction_status: str
    confidence_tier: str | None
    anomaly_level: str | None
    actual_recovered_amount: float
    recovered: bool
    outcome_timestamp: str | None
    # Errors — signed and absolute, computed once here, never fed back into
    # case.prediction or case.recovered_amount.
    probability_error: float  # predicted_probability - (1.0 if recovered else 0.0)
    absolute_probability_error: float
    erv_error: float | None  # predicted_erv - actual_recovered_amount
    absolute_erv_error: float | None
    calibration_bucket: str  # "0.0-0.1", "0.1-0.2", ... "0.9-1.0"


def _calibration_bucket(p: float) -> str:
    p = max(0.0, min(1.0, p))
    lo = min(int(p * 10), 9) / 10.0
    return f"{lo:.1f}-{lo + 0.1:.1f}"


def build_prediction_outcome_records(db: Session) -> list[PredictionOutcomeRecord]:
    """Live-queried every call. Never cached, never persisted back onto a case."""
    cases = db.query(CaseModel).filter(CaseModel.prediction.isnot(None)).all()
    records: list[PredictionOutcomeRecord] = []

    for case in cases:
        pred = case.prediction or {}
        prob = pred.get("recovery_probability")
        if prob is None:
            continue
        pred_ts_raw = pred.get("prediction_timestamp")
        pred_ts = datetime.fromisoformat(pred_ts_raw) if pred_ts_raw else None
        if pred_ts and pred_ts.tzinfo is None:
            pred_ts = pred_ts.replace(tzinfo=UTC)

        outcome_query = db.query(OutcomeEventModel).filter(
            OutcomeEventModel.case_id == case.case_id,
            OutcomeEventModel.match_status == "MATCHED",
        )
        if pred_ts is not None:
            outcome_query = outcome_query.filter(OutcomeEventModel.occurred_at >= pred_ts)
        outcome = outcome_query.order_by(OutcomeEventModel.occurred_at.asc()).first()
        if outcome is None:
            continue  # PENDING - not eligible, never treated as a negative label

        actual_recovered = float(outcome.applied_amount or 0.0)
        recovered = actual_recovered > 0.0
        erv = pred.get("expected_recoverable_value")
        predicted_erv = (
            float(erv)
            if erv is not None
            else (
                float(case.expected_recoverable_value)
                if case.expected_recoverable_value is not None
                else None
            )
        )

        probability_error = float(prob) - (1.0 if recovered else 0.0)
        erv_error = (predicted_erv - actual_recovered) if predicted_erv is not None else None

        records.append(
            PredictionOutcomeRecord(
                case_id=case.case_id,
                risk_category=getattr(case.risk_category, "value", case.risk_category),
                provenance=outcome.provenance,
                decision_timestamp=pred_ts.isoformat() if pred_ts else None,
                predicted_recovery_probability=float(prob),
                predicted_erv=predicted_erv,
                amount_at_risk=float(case.amount_at_risk) if case.amount_at_risk else 0.0,
                model_version=pred.get("model_version", "unknown"),
                prediction_status=pred.get("prediction_status", "unknown"),
                confidence_tier=pred.get("confidence_tier"),
                anomaly_level=(pred.get("anomaly") or {}).get("level"),
                actual_recovered_amount=actual_recovered,
                recovered=recovered,
                outcome_timestamp=outcome.occurred_at.isoformat() if outcome.occurred_at else None,
                probability_error=round(probability_error, 6),
                absolute_probability_error=round(abs(probability_error), 6),
                erv_error=round(erv_error, 2) if erv_error is not None else None,
                absolute_erv_error=round(abs(erv_error), 2) if erv_error is not None else None,
                calibration_bucket=_calibration_bucket(float(prob)),
            )
        )

    return records
