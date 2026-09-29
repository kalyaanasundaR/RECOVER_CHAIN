"""Canonical recovery feature derivation for LIVE inference.

Produces the exact feature vector ml/training/train_recovery_model.py's model was
trained on (FEATURES), from authoritative operational history, with per-feature
provenance. Governing rules (see docs/ai_reconstruction/LIVE_RECOVERY_FEATURE_COVERAGE.md):

* TEMPORAL: every historical feature uses only records strictly BEFORE the prediction
  timestamp (``cutoff``) and strictly before this case was created. Outcomes are dated
  by their own business/audit timestamp, never by the case's *current* state - a prior
  case that recovered after the cutoff is NOT a recovery at prediction time.
* NO FABRICATION: a feature with no real source stays None (status DATA_BLOCKED). A
  feature that has a source but no history yet is COLD_START (documented sentinel or
  None) - never a guessed population value.
* NO SEMANTIC SUBSTITUTION: e.g. "days since RecoverChain first saw the customer" is
  not customer_tenure (account age), so customer_tenure stays DATA_BLOCKED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from ml.inference.model_integrity import CANONICAL_FEATURES

# Single source of truth: ml/inference/model_integrity.py (== training FEATURES, asserted at import
# of the training script and at model load).
CANONICAL_RECOVERY_FEATURES = list(CANONICAL_FEATURES)

# Category vocabularies the trained OneHotEncoder knows (asserted against the artifact in tests).
# A live value outside the vocabulary is NOT passed (it would silently encode as all-zeros,
# indistinguishable from an unseen value): it is left missing instead.
TRAINED_PAYMENT_METHODS = frozenset({"card", "bank_transfer", "wallet", "upi", "unknown"})
TRAINED_CONSENT_STATUSES = frozenset({"ALLOWED", "OPTED_OUT", "UNKNOWN"})

# Cold start (no prior settled history): training used -1.0 for prior_recovery_rate
# ("no history" sentinel, see generate_recovery_outcome). This is a documented
# NO_HISTORY marker, NOT a population prior and NOT customer history.
COLD_START_PRIOR_RECOVERY_RATE = -1.0

# Features with no authoritative source in this application today. Never derived.
DATA_BLOCKED_FEATURES = {
    "customer_tenure": "no account-open date is stored (first-seen-by-RecoverChain is a different quantity)",
    "prior_payment_success_rate": "successful payments are skipped at ingestion (not_at_risk) and never persisted",
    "prior_late_rate": "no per-payment due-date/paid-date history is persisted",
    "invoice_age": "no invoice date is stored separately from the failure event timestamp",
    "customer_engagement": "no engagement telemetry is collected",
    "recent_login": "no login telemetry is collected",
    "previous_contact_success": "no contact-response/reply tracking exists",
    "account_status": "no account-status field exists in any table",
    "historical_recovery_behavior": "training defined it from a latent reliability trait, not an empirical rate",
}


def _naive_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


@dataclass
class FeatureProvenance:
    value: Any
    status: str  # OBSERVED | COLD_START | MISSING | DATA_BLOCKED
    source: str
    cutoff: str | None = None
    records_used: int | None = None
    note: str | None = None

    def as_dict(self) -> dict:
        return {
            "value": self.value,
            "status": self.status,
            "source": self.source,
            "cutoff": self.cutoff,
            "records_used": self.records_used,
            "note": self.note,
        }


@dataclass
class RecoveryFeatureVector:
    values: dict[str, Any]
    provenance: dict[str, FeatureProvenance]
    prediction_timestamp: datetime
    available_count: int = 0
    total_count: int = len(CANONICAL_RECOVERY_FEATURES)
    blocked: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.available_count / self.total_count

    def coverage_label(self) -> str:
        return f"{self.available_count}/{self.total_count}"


def _prior_history(db: Session, customer_id: str, exclude_case_id: str, before: datetime, cutoff):
    """Prior cases of this customer created strictly before ``before`` (and the cutoff),
    each classified using only evidence dated strictly before ``cutoff``."""
    from infrastructure.orm import AuditModel, CaseModel, EventModel, OutcomeEventModel

    prior = (
        db.query(CaseModel)
        .filter(
            CaseModel.customer_id == customer_id,
            CaseModel.case_id != exclude_case_id,
            CaseModel.created_at < before,
            CaseModel.created_at < cutoff,
        )
        .all()
    )
    recovered = not_recovered = 0
    days_to_pay: list[float] = []
    amounts: list[float] = []
    for c in prior:
        amounts.append(float(c.amount_at_risk))
        settled = (
            db.query(OutcomeEventModel)
            .filter(
                OutcomeEventModel.case_id == c.case_id,
                OutcomeEventModel.kind == "SETTLEMENT",
                OutcomeEventModel.effect == "APPLIED",
                OutcomeEventModel.occurred_at < cutoff,
            )
            .order_by(OutcomeEventModel.occurred_at)
            .all()
        )
        applied_total = sum(float(s.applied_amount or 0) for s in settled)
        audits = (
            db.query(AuditModel)
            .filter(AuditModel.case_id == c.case_id, AuditModel.timestamp < cutoff)
            .all()
        )
        states = {a.to_state for a in audits}
        fully = applied_total >= float(c.amount_at_risk) > 0 or "FULLY_RECOVERED" in states
        if fully:
            recovered += 1
            origin = (
                db.query(EventModel.timestamp)
                .filter(EventModel.case_id == c.case_id)
                .order_by(EventModel.timestamp)
                .first()
            )
            if settled and origin:
                days_to_pay.append(
                    max(0.0, (settled[0].occurred_at - origin[0]).total_seconds() / 86400)
                )
        elif "CLOSED_NOT_RECOVERED" in states:
            not_recovered += 1
        # PARTIALLY_RECOVERED / still open / cancelled / denied / stopped: not counted either way.
    return {
        "n_prior_cases": len(prior),
        "recovered": recovered,
        "not_recovered": not_recovered,
        "days_to_pay": days_to_pay,
        "amounts": amounts,
    }


def derive_recovery_features(
    db: Session, case_id: str, prediction_timestamp: datetime | None = None
) -> RecoveryFeatureVector:
    from application.contact_permissions import PermissionChannel, resolve_all
    from infrastructure.orm import (
        CaseModel,
        ContactPermissionModel,
        EventModel,
        ExecutionAttemptModel,
    )

    now = _naive_utc(prediction_timestamp) or _naive_utc(datetime.now(UTC))
    cutoff_s = now.isoformat()
    values: dict[str, Any] = dict.fromkeys(CANONICAL_RECOVERY_FEATURES)
    prov: dict[str, FeatureProvenance] = {}

    case = db.query(CaseModel).filter(CaseModel.case_id == case_id).first()
    if case is None:
        raise LookupError(case_id)

    def put(name, value, status, source, records=None, note=None):
        values[name] = value
        prov[name] = FeatureProvenance(value, status, source, cutoff_s, records, note)

    # -- direct case facts --------------------------------------------------------
    put(
        "amount_at_risk", float(case.amount_at_risk), "OBSERVED", "recovery_cases.amount_at_risk", 1
    )
    put(
        "domain",
        getattr(case.risk_category, "value", str(case.risk_category)),
        "OBSERVED",
        "recovery_cases.risk_category",
        1,
    )

    origin_row = (
        db.query(EventModel.timestamp)
        .filter(EventModel.case_id == case_id)
        .order_by(EventModel.timestamp)
        .first()
    )
    if origin_row and origin_row[0] <= now:
        age = max(0, (now - _naive_utc(origin_row[0])).days)
        put(
            "age_of_obligation",
            age,
            "OBSERVED",
            "revenue_events.timestamp (earliest event of this case)",
            1,
        )
    else:
        put(
            "age_of_obligation",
            None,
            "MISSING",
            "revenue_events.timestamp",
            0,
            "no event timestamp at or before the cutoff",
        )

    # -- rolling history (strictly before this case AND the cutoff) -----------------
    before = _naive_utc(case.created_at) or now
    hist = _prior_history(db, case.customer_id, case_id, before, now)
    settled_n = hist["recovered"] + hist["not_recovered"]
    src_hist = "recovery_cases + outcome_events(SETTLEMENT,APPLIED) + audit_records, dated < cutoff"
    if hist["n_prior_cases"] == 0 or settled_n == 0:
        put(
            "number_of_previous_recoveries",
            0,
            "COLD_START",
            src_hist,
            hist["n_prior_cases"],
            "no settled prior obligation",
        )
        put(
            "number_of_previous_failed_attempts",
            0,
            "COLD_START",
            src_hist,
            hist["n_prior_cases"],
            "no settled prior obligation",
        )
        put(
            "prior_recovery_rate",
            COLD_START_PRIOR_RECOVERY_RATE,
            "COLD_START",
            src_hist,
            hist["n_prior_cases"],
            "NO_HISTORY sentinel (-1.0), same as training; not a population prior",
        )
    else:
        put("number_of_previous_recoveries", hist["recovered"], "OBSERVED", src_hist, settled_n)
        put(
            "number_of_previous_failed_attempts",
            hist["not_recovered"],
            "OBSERVED",
            src_hist,
            settled_n,
        )
        put("prior_recovery_rate", hist["recovered"] / settled_n, "OBSERVED", src_hist, settled_n)

    if hist["amounts"]:
        avg_amt = sum(hist["amounts"]) / len(hist["amounts"])
        put(
            "amount_vs_customer_average",
            float(case.amount_at_risk) / max(avg_amt, 1.0),
            "OBSERVED",
            "recovery_cases.amount_at_risk (mean of prior cases)",
            len(hist["amounts"]),
            "rolling mean of the customer's earlier obligations",
        )
    else:
        put(
            "amount_vs_customer_average",
            None,
            "COLD_START",
            "recovery_cases.amount_at_risk",
            0,
            "no earlier obligation to average; left missing",
        )

    if hist["days_to_pay"]:
        put(
            "average_days_to_payment",
            sum(hist["days_to_pay"]) / len(hist["days_to_pay"]),
            "OBSERVED",
            "outcome_events.occurred_at - revenue_events.timestamp (prior recovered cases)",
            len(hist["days_to_pay"]),
        )
    else:
        put(
            "average_days_to_payment",
            None,
            "COLD_START",
            "outcome_events",
            0,
            "no prior recovered obligation; left missing",
        )

    # -- contact history / permissions ---------------------------------------------
    n_comm = (
        db.query(ExecutionAttemptModel)
        .join(CaseModel, CaseModel.case_id == ExecutionAttemptModel.case_id)
        .filter(
            CaseModel.customer_id == case.customer_id,
            ExecutionAttemptModel.channel == "communication",
            ExecutionAttemptModel.created_at < now,
        )
        .count()
    )
    put(
        "reminder_history",
        n_comm,
        "OBSERVED",
        "execution_attempts(channel=communication) < cutoff",
        n_comm,
        "0 is an observed absence of communication attempts",
    )

    perms = (
        db.query(ContactPermissionModel)
        .filter(ContactPermissionModel.customer_id == case.customer_id)
        .all()
    )
    resolved = resolve_all(perms, at=now.replace(tzinfo=UTC), known_at=now.replace(tzinfo=UTC))
    pay = resolved[PermissionChannel.PAYMENT]
    suppressed = any(r.status == "SUPPRESSED" for r in resolved.values())
    if pay.status in TRAINED_CONSENT_STATUSES:
        put(
            "consent_status",
            pay.status,
            "OBSERVED",
            "contact_permissions (resolve_all, bitemporal at cutoff)",
            len(perms),
            "UNKNOWN = no permission record; observed fact, not consent",
        )
    else:
        put(
            "consent_status",
            None,
            "MISSING",
            "contact_permissions",
            len(perms),
            f"{pay.status} is outside the trained vocabulary; carried by suppression_status instead",
        )
    put(
        "suppression_status",
        int(suppressed),
        "OBSERVED",
        "contact_permissions (resolve_all)",
        len(perms),
    )

    # -- partially available (only when positive evidence exists) --------------------
    raw = {}
    if origin_row is not None:
        ev = (
            db.query(EventModel)
            .filter(EventModel.case_id == case_id)
            .order_by(EventModel.timestamp)
            .first()
        )
        raw = {str(k).lower(): v for k, v in (ev.raw_payload or {}).items()} if ev else {}
    pm = raw.get("payment_method") or raw.get("method")
    pm_norm = str(pm).strip().lower() if pm not in (None, "") else None
    if pm_norm in TRAINED_PAYMENT_METHODS:
        put("payment_method", pm_norm, "OBSERVED", "revenue_events.raw_payload.payment_method", 1)
    elif pm_norm is not None:
        put(
            "payment_method",
            None,
            "MISSING",
            "revenue_events.raw_payload",
            1,
            f"{pm_norm!r} is outside the trained vocabulary; left missing, not mapped",
        )
    else:
        put(
            "payment_method",
            None,
            "MISSING",
            "revenue_events.raw_payload",
            0,
            "column not in this upload",
        )

    signals = (
        (case.diagnosis or {}).get("supporting_signals", {})
        if isinstance(case.diagnosis, dict)
        else {}
    )
    if signals.get("has_expired_card"):
        put(
            "payment_method_valid",
            0,
            "OBSERVED",
            "recovery_cases.diagnosis.supporting_signals.has_expired_card",
            1,
            "only positive evidence of an invalid method is recorded; absence is left missing",
        )
    else:
        put(
            "payment_method_valid",
            None,
            "MISSING",
            "recovery_cases.diagnosis",
            0,
            "no positive evidence either way",
        )
    if signals.get("has_dispute"):
        put(
            "dispute_indicator",
            1,
            "OBSERVED",
            "recovery_cases.diagnosis.supporting_signals.has_dispute",
            1,
            "only positive evidence is recorded; absence is left missing",
        )
    else:
        put(
            "dispute_indicator",
            None,
            "MISSING",
            "recovery_cases.diagnosis",
            0,
            "no positive evidence either way",
        )

    # -- data-blocked ------------------------------------------------------------------
    blocked = []
    for name, why in DATA_BLOCKED_FEATURES.items():
        values[name] = None
        prov[name] = FeatureProvenance(None, "DATA_BLOCKED", "none", cutoff_s, 0, why)
        blocked.append(name)

    available = sum(
        1 for p in prov.values() if p.status in ("OBSERVED", "COLD_START") and p.value is not None
    )
    return RecoveryFeatureVector(
        values=values,
        provenance=prov,
        prediction_timestamp=now,
        available_count=available,
        blocked=blocked,
    )
