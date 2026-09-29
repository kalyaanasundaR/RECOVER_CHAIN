"""Canonical ingestion contract.

Every input source normalises into the same canonical ``RevenueEvent`` through the
helpers in this module, so the decision engine never has to understand an external
schema and every source is held to the same validation rules:

    external source
      -> source adapter      (an /events request body, or a mapped dataset row)
      -> validate + normalise (this module)
      -> canonical RevenueEvent carrying provenance
      -> deduplication on (external_system, external_event_id)
      -> CaseEngine

Adding a new source (CSV / JSON / webhook / database connector) means writing only
the adapter step: turn the source's record into these fields and call the same
validators. Nothing downstream changes.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any


class SourceType:
    """Where an event entered the system. Stored on every event row."""

    API = "API"  # POST /events
    API_BATCH = "API_BATCH"  # POST /events/batch
    FILE_UPLOAD = "FILE_UPLOAD"  # dataset rows replayed via generate-cases


# Events replayed from an uploaded file all share one external system. Their identity
# is derived from row CONTENT (see event_fingerprint), not from the upload, so the same
# real event arriving in two uploads - or twice in one upload by mistake - is recognised
# as the same event instead of silently becoming a second case.
FILE_EXTERNAL_SYSTEM = "file_upload"

# recovery_cases.amount_at_risk is Numeric(18, 4): it cannot hold 1e14 or more. Keep a
# wide margin below that so a single bad value cannot overflow the column on PostgreSQL
# or distort every financial aggregate (a 1e15 amount previously did exactly that).
MAX_AMOUNT = Decimal("10000000000000")  # 1e13

# Active ISO 4217 alphabetic codes. Validation only - no conversion is performed
# anywhere in RecoverChain (multi-currency arithmetic is not implemented).
ISO_4217 = frozenset(
    """
    AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BRL BSD
    BTN BWP BYN BZD CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB
    EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK
    JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA
    MKD MMK MNT MOP MRU MUR MVR MWK MXN MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK
    PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN
    SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD UYU UZS VES VND VUV WST
    XAF XCD XOF XPF YER ZAR ZMW ZWG
    """.split()
)


class RejectReason:
    """Stable, machine-readable reasons a record was not accepted."""

    MISSING_AMOUNT = "missing_amount"
    INVALID_AMOUNT = "invalid_amount"
    NON_POSITIVE_AMOUNT = "non_positive_amount"
    AMOUNT_TOO_LARGE = "amount_too_large"
    INVALID_CURRENCY = "invalid_currency"
    MISSING_IDENTIFIER = "missing_identifier"
    INVALID_TIMESTAMP = "invalid_timestamp"
    MISSING_TARGET = "missing_target"
    AMBIGUOUS_TARGET = "ambiguous_target"
    NOT_AT_RISK = "not_at_risk"  # the row records a successful payment
    DUPLICATE_EVENT = "duplicate_event"
    CASE_CLOSED = "case_closed"
    MISSING_FIELD = "missing_field"  # a required request field is absent
    INVALID_VALUE = "invalid_value"  # e.g. an unknown risk_category
    PROCESSING_ERROR = "processing_error"  # valid record that failed while being stored
    # the row's category column does not name a known domain (Phase 4). Never defaulted.
    UNRESOLVED_DOMAIN = "unresolved_domain"


class ValidationFailure(ValueError):
    """A record failed the ingestion contract. ``reason`` is a RejectReason code."""

    def __init__(self, reason: str, field_name: str, detail: str):
        super().__init__(f"{field_name}: {detail}")
        self.reason = reason
        self.field_name = field_name
        self.detail = detail


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:  # pandas NaT / NA without importing pandas here
        return bool(value != value)
    except Exception:
        return False


def normalize_identifier(value: Any, field_name: str) -> str:
    """A non-blank identifier, surrounding whitespace removed."""
    if _is_missing(value):
        raise ValidationFailure(RejectReason.MISSING_IDENTIFIER, field_name, "is required")
    text = str(value).strip()
    if not text:
        raise ValidationFailure(RejectReason.MISSING_IDENTIFIER, field_name, "is required")
    return text


def parse_amount(value: Any, field_name: str = "amount") -> Decimal:
    """A strictly positive, finite amount below MAX_AMOUNT.

    Deliberately strict: grouped or symbol-prefixed strings such as "1,200" or
    "$45.10" are rejected rather than guessed at, because "1.200,50" means different
    things in different locales. A rejected value is reported, never silently coerced.
    """
    if _is_missing(value):
        raise ValidationFailure(RejectReason.MISSING_AMOUNT, field_name, "is required")
    if isinstance(value, bool):
        raise ValidationFailure(RejectReason.INVALID_AMOUNT, field_name, "must be a number")
    try:
        amount = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ValidationFailure(
            RejectReason.INVALID_AMOUNT, field_name, f"{value!r} is not a number"
        ) from None
    if not amount.is_finite():
        raise ValidationFailure(RejectReason.INVALID_AMOUNT, field_name, "must be finite")
    if amount <= 0:
        raise ValidationFailure(
            RejectReason.NON_POSITIVE_AMOUNT, field_name, "must be greater than zero"
        )
    if amount >= MAX_AMOUNT:
        raise ValidationFailure(
            RejectReason.AMOUNT_TOO_LARGE, field_name, f"must be below {MAX_AMOUNT}"
        )
    return amount


def normalize_currency(value: Any, default: str | None = None) -> str:
    """An upper-cased, active ISO 4217 code ("usd" -> "USD")."""
    if _is_missing(value):
        if default is None:
            raise ValidationFailure(RejectReason.INVALID_CURRENCY, "currency", "is required")
        value = default
    code = str(value).strip().upper()
    if code not in ISO_4217:
        raise ValidationFailure(
            RejectReason.INVALID_CURRENCY, "currency", f"{value!r} is not an ISO 4217 code"
        )
    return code


def parse_timestamp(value: Any, field_name: str = "timestamp") -> datetime:
    """A timezone-aware datetime. Naive values are taken as UTC."""
    if _is_missing(value):
        raise ValidationFailure(RejectReason.INVALID_TIMESTAMP, field_name, "is required")
    if isinstance(value, datetime):
        ts = value
    else:
        try:
            from dateutil.parser import parse

            ts = parse(str(value))
        except (ValueError, OverflowError, TypeError):
            raise ValidationFailure(
                RejectReason.INVALID_TIMESTAMP, field_name, f"{value!r} is not a date"
            ) from None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def event_fingerprint(
    *,
    customer_id: str,
    risk_category: str,
    amount: Decimal,
    currency: str,
    timestamp: datetime,
    reference_id: str | None,
    failure_code: str | None,
) -> str:
    """Deterministic identity of an event, derived from what the event says.

    Used when a source provides no external event id of its own. Two records that
    describe the same failure produce the same fingerprint regardless of which file,
    upload, row position or ingestion run they arrived in.
    """
    canonical = "|".join(
        [
            customer_id,
            str(getattr(risk_category, "value", risk_category)),
            f"{amount.quantize(Decimal('0.0001'))}",
            currency,
            timestamp.astimezone(UTC).isoformat(),
            reference_id or "",
            (failure_code or "").strip().lower(),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def fingerprint_event_id(fingerprint: str, occurrence: int) -> str:
    """External event id for a fingerprinted record.

    ``occurrence`` numbers genuinely identical records within ONE source file
    (0, 1, 2, ...), so two real, indistinguishable failures in the same file are both
    kept, while re-uploading that file maps every record back onto the same ids.
    """
    return f"fp:{fingerprint}:{occurrence}"


# -- identity v2 ----------------------------------------------------------------------
#
# v1 (above) hashes the risk category into a file row's identity. Since Phase 4 the
# category of a file row is a CLASSIFICATION that can legitimately change - a dataset's
# domain can be declared after an earlier run inferred another one - and the same real
# row must remain the same event when it does. v2 hashes only what the row itself says.
# Rows already stored under a v1 id are still recognised (legacy_v1_event_ids), and
# events stored before content identity existed are re-identified by
# application/identity_reconciliation.py.


def event_fingerprint_v2(
    *,
    customer_id: str,
    amount: Decimal,
    currency: str,
    timestamp: datetime,
    reference_id: str | None,
    failure_code: str | None,
) -> str:
    canonical = "|".join(
        [
            "v2",
            customer_id,
            f"{amount.quantize(Decimal('0.0001'))}",
            currency,
            timestamp.astimezone(UTC).isoformat(),
            reference_id or "",
            (failure_code or "").strip().lower(),
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def fingerprint_event_id_v2(fingerprint: str, occurrence: int) -> str:
    return f"fp2:{fingerprint}:{occurrence}"


def legacy_v1_event_ids(
    *,
    customer_id: str,
    amount: Decimal,
    currency: str,
    timestamp: datetime,
    reference_id: str | None,
    failure_code: str | None,
    occurrence: int,
) -> list[str]:
    """Every v1 id the same row could already be stored under.

    v1 included the category the retired column-name heuristic assigned, which cannot be
    recomputed, so one id per category is produced. Phase 3 also turned a blank reason
    cell into the failure code "nan"; that variant is included for rows with no code.
    """
    from domain.models import RiskCategory

    codes = [failure_code] if failure_code else [None, "nan"]
    return [
        fingerprint_event_id(
            event_fingerprint(
                customer_id=customer_id,
                risk_category=category,
                amount=amount,
                currency=currency,
                timestamp=timestamp,
                reference_id=reference_id,
                failure_code=code,
            ),
            occurrence,
        )
        for category in RiskCategory
        for code in codes
    ]


_FAILURE_CODE_KEY_TOKENS = ("reason", "cause", "failure_code", "decline", "error_code")


def derive_failure_code(payload: dict) -> str | None:
    """The failure code a file row carries (for diagnosis and for identity).

    An explicit ``failure_code`` key wins; otherwise the first column whose name contains
    one of the tokens above and holds a value. ``payload`` must be JSON-safe (missing
    cells are None). Shared by ingestion and identity reconciliation so both derive the
    same code from the same row. A blank cell is no code - it used to become "nan".
    """
    explicit = payload.get("failure_code")
    if not _is_missing(explicit):
        return str(explicit).strip().lower()
    for key, value in payload.items():
        name = str(key).lower()
        if not _is_missing(value) and any(tok in name for tok in _FAILURE_CODE_KEY_TOKENS):
            return str(value).strip().lower()
    return None


def new_run_id() -> str:
    return f"ing_{uuid.uuid4().hex[:16]}"


@dataclass
class IngestionRunReport:
    """In-memory tally for one ingestion run, persisted as an IngestionRunModel row.

    Kept in memory and written once, in the same database session as the run's own
    writes: on success it is committed together with them, on failure the run's
    writes are rolled back and a FAILED record is written instead. That avoids a
    second concurrent write connection, which SQLite would serialise into a lock
    timeout.
    """

    source_type: str
    source_ref: str | None = None
    source_name: str | None = None
    run_id: str = field(default_factory=new_run_id)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    rows_in_source: int | None = None
    rows_received: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0
    rows_duplicate: int = 0
    rows_not_at_risk: int = 0
    rows_not_processed: int = 0
    cases_created: int = 0
    cases_updated: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    rejected_samples: list[dict] = field(default_factory=list)
    # How the run's rows were assigned a domain (file sources), and what it produced.
    domain_classification: dict | None = None
    accepted_by_domain: dict[str, int] = field(default_factory=dict)

    MAX_SAMPLES = 50

    def reject(self, reason: str, *, row=None, field_name=None, detail=None, also=()):
        """Count ONE rejected record. ``also`` lists further reasons the same record
        failed, so rejection_counts may add up to more than rows_rejected."""
        self.rows_rejected += 1
        for r in (reason, *also):
            self.rejection_counts[r] = self.rejection_counts.get(r, 0) + 1
        if len(self.rejected_samples) < self.MAX_SAMPLES:
            self.rejected_samples.append(
                {"row": row, "reason": reason, "field": field_name, "detail": detail}
            )

    def to_model(self, status: str, error_message: str | None = None):
        from infrastructure.orm import IngestionRunModel

        return IngestionRunModel(
            run_id=self.run_id,
            source_type=self.source_type,
            source_ref=self.source_ref,
            source_name=self.source_name,
            status=status,
            started_at=self.started_at,
            completed_at=datetime.now(UTC),
            rows_in_source=self.rows_in_source,
            rows_received=self.rows_received,
            rows_accepted=self.rows_accepted,
            rows_rejected=self.rows_rejected,
            rows_duplicate=self.rows_duplicate,
            rows_not_at_risk=self.rows_not_at_risk,
            rows_not_processed=self.rows_not_processed,
            cases_created=self.cases_created,
            cases_updated=self.cases_updated,
            rejection_counts=self.rejection_counts,
            rejected_samples=self.rejected_samples,
            domain_classification=(
                {
                    "classification": self.domain_classification,
                    "accepted_by_domain": self.accepted_by_domain,
                }
                if self.domain_classification is not None
                else None
            ),
            error_message=error_message,
        )


def ingestion_run_to_dict(run) -> dict:
    """Serialise an IngestionRunModel row for the API."""
    return {
        "run_id": run.run_id,
        "source_type": run.source_type,
        "source_ref": run.source_ref,
        "source_name": run.source_name,
        "status": run.status,
        "started_at": run.started_at.isoformat() if run.started_at else "",
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "rows_in_source": run.rows_in_source,
        "rows_received": run.rows_received or 0,
        "rows_accepted": run.rows_accepted or 0,
        "rows_rejected": run.rows_rejected or 0,
        "rows_duplicate": run.rows_duplicate or 0,
        "rows_not_at_risk": run.rows_not_at_risk or 0,
        "rows_not_processed": run.rows_not_processed or 0,
        "cases_created": run.cases_created or 0,
        "cases_updated": run.cases_updated or 0,
        "rejection_counts": run.rejection_counts or {},
        "rejected_samples": run.rejected_samples or [],
        "domain_classification": run.domain_classification,
        "error_message": run.error_message,
    }
