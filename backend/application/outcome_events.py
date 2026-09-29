"""Authoritative outcome events: money received, an obligation cancelled, or a
promise-to-pay created / broken.

    external source
      -> validate + normalise    (this module, reusing application/ingestion.py)
      -> match to an obligation  (domain + customer + reference; never guessed)
      -> reconcile against the matched case's financial position
      -> persist (append-only outcome_events row + the case's updated position)
      -> audit

An outcome event is applied EXACTLY ONCE, at ingestion, in the order it is received.
Its match and its financial effect are decided then and never revisited: a later,
out-of-order (small ``occurred_at``) event is simply applied against whatever the
obligation's position is WHEN IT ARRIVES. This is deliberate - RecoverChain does not
implement event-sourcing / retroactive recomputation, so an amount already reported as
recovered is never rewritten by a later-discovered earlier event. ``occurred_at`` is
kept for audit, for ordering promise-to-pay derivations, and to detect "arrived before
the obligation existed" (see rematch_unmatched_outcomes for the one explicit, always
manual way that is revisited).

ACTION ATTEMPT != ACTION SUCCESS != REVENUE RECOVERED: nothing in this module reads
ExecutionRecord or ActionType. The only inputs to a financial decision are the events
here and the ones already applied to the same case.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from application.evidence_sources import is_usable
from application.ingestion import (
    RejectReason as IngestionRejectReason,
)
from application.ingestion import (
    ValidationFailure,
    normalize_currency,
    normalize_identifier,
    parse_amount,
    parse_timestamp,
)
from application.outcome_provenance import (
    SOURCE_GATED_VALUES,
    OutcomeProvenance,
    normalize_provenance,
)
from domain.lifecycle import OUTCOME_FINAL_STATES, CaseClosedError, CaseLifecycleManager
from domain.models import CaseState, Money, RecoveryOutcome, RecoveryOutcomeStatus, RiskCategory
from domain.revenue_domains import OutcomeKind, domain_spec, outcome_type_spec
from infrastructure.orm import CustomerIdentityLinkModel, OutcomeEventModel, PromiseModel

Q4 = Decimal("0.0001")


class OutcomeRejectReason:
    """Outcome-specific reasons, alongside the shared application.ingestion.RejectReason
    codes (missing_identifier, invalid_amount, invalid_currency, invalid_timestamp...)."""

    MISSING_DOMAIN = "missing_domain"
    INVALID_DOMAIN = "invalid_domain"
    UNSUPPORTED_OUTCOME_TYPE = "unsupported_outcome_type"
    OUTCOME_TYPE_DOMAIN_MISMATCH = "outcome_type_domain_mismatch"
    INVALID_PROVENANCE = "invalid_provenance"


class MatchReason:
    """Why an outcome event could not be matched to an obligation. Never guessed:
    each of these means a deterministic rule found no safe target."""

    NO_REFERENCE = "NO_REFERENCE"  # no reference_id: an obligation cannot be identified
    NO_OBLIGATION = "NO_OBLIGATION"  # no case exists yet for this domain+customer+reference
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"  # differs from the case's currency; never converted
    UNLINKED_CUSTOMER_IDENTITY = "UNLINKED_CUSTOMER_IDENTITY"  # customer_system given, no link


class Effect:
    APPLIED = "APPLIED"  # changed the case's recovered / cancelled amount
    EXCESS = "EXCESS"  # money arrived beyond what was outstanding; recorded, not recovered
    NO_EFFECT = "NO_EFFECT"  # e.g. a cancellation on an obligation already at zero outstanding


@dataclass
class OutcomeEventInput:
    external_system: str
    external_event_id: str
    domain: str
    outcome_type: str
    customer_id: str
    reference_id: str | None = None
    customer_system: str | None = None
    amount: Any = None
    currency: str | None = None
    occurred_at: Any = None
    promised_date: Any = None
    raw_payload: dict = field(default_factory=dict)
    source_type: str | None = None
    source_ref: str | None = None
    ingestion_run_id: str | None = None
    # Phase 8A: where this outcome actually came from (application.outcome_provenance).
    # None -> normalised to TEST, never to anything that reads as authoritative
    # production evidence.
    provenance: str | None = None
    # Phase 9: required for provenance=human_confirmed (must point outside
    # RecoverChain's own generated ids); required, and checked against the
    # evidence-source registry, for the per-integration provenances.
    evidence_reference: str | None = None
    source_id: str | None = None


def _quant(d: Decimal) -> Decimal:
    return d.quantize(Q4)


# Prefixes every id RecoverChain itself generates. A human-confirmed evidence
# reference matching one of these is presumptively self-referential (pointing back
# at RecoverChain's own record instead of something external) and is rejected -
# Phase 9 Step 6: "a user must NOT be able to turn a RecoverChain test outcome into
# human-confirmed evidence merely by clicking a button."
_RECOVERCHAIN_ID_PREFIXES = (
    "oc_",
    "case_",
    "cobs_",
    "evt_",
    "out_",
    "src_",
    "snap_",
    "pol_",
    "exec_",
    "diag_",
    "pred_",
    "ing_",
    "perm_",
    "prm_",
)


def _looks_like_recoverchain_id(value: str) -> bool:
    lowered = value.strip().lower()
    return any(lowered.startswith(p) for p in _RECOVERCHAIN_ID_PREFIXES)


def validate_outcome_item(item: OutcomeEventInput, db) -> dict:
    """The shared validation contract, reused from application.ingestion. Returns a dict
    of normalised fields. Raises ValidationFailure on the first problem found.

    `db` is required (not optional) so the Phase 9 evidence-integrity rules can never
    be silently skipped by a caller forgetting to wire them up: a human_confirmed
    claim must carry a genuine external evidence_reference, and a source-gated
    provenance (production/payment_provider/...) must name a registered, verified,
    enabled evidence source of the matching type and domain - both checked here,
    against the real database, not assumed from the request's own say-so."""
    external_system = normalize_identifier(item.external_system, "external_system")
    external_event_id = normalize_identifier(item.external_event_id, "external_event_id")

    if not item.domain:
        raise ValidationFailure(OutcomeRejectReason.MISSING_DOMAIN, "domain", "is required")
    try:
        domain = RiskCategory(item.domain)
    except ValueError:
        raise ValidationFailure(
            OutcomeRejectReason.INVALID_DOMAIN, "domain", f"{item.domain!r} is not a known domain"
        ) from None

    if not item.outcome_type:
        raise ValidationFailure(IngestionRejectReason.MISSING_FIELD, "outcome_type", "is required")
    outcome_type = str(item.outcome_type).strip().upper().replace(" ", "_").replace("-", "_")
    spec = outcome_type_spec(outcome_type)
    if spec is None:
        raise ValidationFailure(
            OutcomeRejectReason.UNSUPPORTED_OUTCOME_TYPE,
            "outcome_type",
            f"{item.outcome_type!r} is not a recognised outcome type",
        )
    kind, expected_domain = spec
    if expected_domain != domain:
        raise ValidationFailure(
            OutcomeRejectReason.OUTCOME_TYPE_DOMAIN_MISMATCH,
            "outcome_type",
            f"{outcome_type} is not valid for domain {domain.value} "
            f"(it belongs to {expected_domain.value})",
        )

    customer_id = normalize_identifier(item.customer_id, "customer_id")
    customer_system = None
    if item.customer_system is not None and str(item.customer_system).strip():
        customer_system = str(item.customer_system).strip()

    reference_id = None
    if item.reference_id is not None and str(item.reference_id).strip():
        reference_id = str(item.reference_id).strip()

    occurred_at = parse_timestamp(item.occurred_at, "occurred_at")

    amount = currency = None
    if kind in (OutcomeKind.SETTLEMENT, OutcomeKind.PROMISE_CREATED):
        amount = parse_amount(item.amount, "amount")
        currency = normalize_currency(item.currency)

    promised_date = None
    if kind == OutcomeKind.PROMISE_CREATED:
        promised_date = parse_timestamp(item.promised_date, "promised_date")

    try:
        provenance = normalize_provenance(item.provenance)
    except ValueError as e:
        raise ValidationFailure(
            OutcomeRejectReason.INVALID_PROVENANCE, "provenance", str(e)
        ) from None

    evidence_reference = (item.evidence_reference or "").strip() or None
    if provenance == OutcomeProvenance.HUMAN_CONFIRMED:
        if not evidence_reference:
            raise ValidationFailure(
                OutcomeRejectReason.INVALID_PROVENANCE,
                "evidence_reference",
                "human_confirmed requires evidence_reference - a pointer to something "
                "outside RecoverChain (a ticket id, a call-log reference, an email "
                "thread) - so a test outcome can never become 'confirmed' evidence "
                "merely by being reposted with a different provenance label",
            )
        if _looks_like_recoverchain_id(evidence_reference):
            raise ValidationFailure(
                OutcomeRejectReason.INVALID_PROVENANCE,
                "evidence_reference",
                f"{evidence_reference!r} looks like a RecoverChain-generated id, not "
                "external evidence - human confirmation must point outside the system "
                "it is confirming",
            )
    elif provenance in SOURCE_GATED_VALUES:
        if not is_usable(db, item.source_id, source_type=provenance, domain=domain.value):
            raise ValidationFailure(
                OutcomeRejectReason.INVALID_PROVENANCE,
                "source_id",
                f"{provenance!r} requires a registered, VERIFIED, ENABLED evidence "
                f"source of type {provenance!r} for domain {domain.value!r} "
                f"(source_id={item.source_id!r}) - no such source currently exists in "
                "this deployment",
            )

    return {
        "external_system": external_system,
        "external_event_id": external_event_id,
        "domain": domain,
        "outcome_type": outcome_type,
        "kind": kind,
        "customer_id": customer_id,
        "customer_system": customer_system,
        "reference_id": reference_id,
        "amount": amount,
        "currency": currency,
        "occurred_at": occurred_at,
        "promised_date": promised_date,
        "provenance": provenance,
        "evidence_reference": evidence_reference,
        "source_id": item.source_id,
    }


def resolve_customer_id(
    db, customer_id: str, customer_system: str | None
) -> tuple[str | None, str]:
    """(resolved_customer_id, match_reason_if_unresolved-or-''). No customer_system given
    -> the id is used as given (RecoverChain's own id). A customer_system is honoured ONLY
    through an explicit, previously-recorded link - never inferred."""
    if not customer_system:
        return customer_id, ""
    link = (
        db.query(CustomerIdentityLinkModel)
        .filter(
            CustomerIdentityLinkModel.system == customer_system,
            CustomerIdentityLinkModel.external_customer_id == customer_id,
        )
        .first()
    )
    if link is None:
        return None, MatchReason.UNLINKED_CUSTOMER_IDENTITY
    return link.customer_id, ""


def find_matching_case(
    case_repo, *, domain: RiskCategory, customer_id: str, reference_id: str | None
):
    """(case_or_None, match_reason_if_none). The obligation is domain + customer +
    reference (domain.revenue_domains.obligation_key) - the same identity correlation
    uses. Every case for the obligation is a candidate, whatever its state: a settlement
    can legitimately arrive for an obligation whose case already closed. Prefer the most
    recently created case that can still receive money (not yet FULLY_RECOVERED or
    CANCELLED); if every case for the obligation is already fully settled or cancelled,
    the most recent one is still returned so the event can be recorded as EXCESS against
    real history, rather than reported as unmatched."""
    if not reference_id:
        return None, MatchReason.NO_REFERENCE
    candidates = case_repo.get_cases_for_obligation(customer_id, domain, reference_id)
    if not candidates:
        return None, MatchReason.NO_OBLIGATION
    open_ones = [c for c in candidates if c.current_state not in OUTCOME_FINAL_STATES]
    return (open_ones[-1] if open_ones else candidates[-1]), ""


@dataclass
class ReconciliationResult:
    new_recovered: Decimal
    new_outstanding: Decimal
    new_cancelled: Decimal
    applied_amount: Decimal
    excess_amount: Decimal
    effect: str
    new_case_state: CaseState
    new_outcome_status: RecoveryOutcomeStatus


def reconcile_position(
    *,
    kind: str,
    amount: Decimal | None,
    current_state: CaseState,
    recovered: Decimal,
    outstanding: Decimal,
    cancelled: Decimal,
) -> ReconciliationResult:
    """Pure function: the case's position after one outcome event is applied. No I/O, no
    side effects - easy to test exhaustively for duplicate / out-of-order / overpayment /
    post-closure sequences. Invariant maintained on every call:
    recovered + outstanding + cancelled == recovered_before + outstanding_before +
    cancelled_before (money only ever moves between these three buckets, or arrives as
    excess which touches none of them)."""
    already_final = current_state in OUTCOME_FINAL_STATES

    if kind == OutcomeKind.SETTLEMENT:
        amount = amount or Decimal(0)
        if already_final or outstanding <= 0:
            applied = Decimal(0)
            excess = amount
            effect = Effect.EXCESS if amount > 0 else Effect.NO_EFFECT
            new_recovered, new_outstanding, new_cancelled = recovered, outstanding, cancelled
        else:
            applied = min(amount, outstanding)
            excess = amount - applied
            new_recovered = _quant(recovered + applied)
            new_outstanding = _quant(outstanding - applied)
            new_cancelled = cancelled
            effect = Effect.APPLIED
        new_state = (
            CaseState.FULLY_RECOVERED
            if new_outstanding <= 0 and new_recovered > 0
            else (CaseState.PARTIALLY_RECOVERED if effect == Effect.APPLIED else current_state)
        )
        new_status = (
            RecoveryOutcomeStatus.FULLY_RECOVERED
            if new_state == CaseState.FULLY_RECOVERED
            else RecoveryOutcomeStatus.PARTIALLY_RECOVERED
            if new_state == CaseState.PARTIALLY_RECOVERED
            else RecoveryOutcomeStatus.PENDING_VERIFICATION
        )

    elif kind == OutcomeKind.CANCELLATION:
        applied = excess = Decimal(0)
        if already_final or outstanding <= 0:
            effect = Effect.NO_EFFECT
            new_recovered, new_outstanding, new_cancelled = recovered, outstanding, cancelled
            new_state = current_state
        else:
            new_cancelled = _quant(cancelled + outstanding)
            new_outstanding = Decimal("0.0000")
            new_recovered = recovered
            effect = Effect.APPLIED
            new_state = CaseState.CANCELLED
        new_status = (
            RecoveryOutcomeStatus.CANCELLED
            if new_state == CaseState.CANCELLED
            else RecoveryOutcomeStatus.PENDING_VERIFICATION
        )

    else:  # PROMISE_CREATED / PROMISE_BROKEN never move money or case state
        applied = excess = Decimal(0)
        effect = Effect.NO_EFFECT
        new_recovered, new_outstanding, new_cancelled = recovered, outstanding, cancelled
        new_state = current_state
        new_status = RecoveryOutcomeStatus.PENDING_VERIFICATION

    return ReconciliationResult(
        new_recovered=new_recovered,
        new_outstanding=new_outstanding,
        new_cancelled=new_cancelled,
        applied_amount=_quant(applied),
        excess_amount=_quant(excess),
        effect=effect,
        new_case_state=new_state,
        new_outcome_status=new_status,
    )


def _decimals(case) -> tuple[Decimal, Decimal, Decimal]:
    recovered = case.recovered_amount.amount if case.recovered_amount else Decimal(0)
    outstanding = (
        case.outstanding_amount.amount
        if case.outstanding_amount is not None
        else case.amount_at_risk.amount
    )
    cancelled = case.cancelled_amount.amount if case.cancelled_amount else Decimal(0)
    return recovered, outstanding, cancelled


def _reason_text(kind: str, result: ReconciliationResult) -> str:
    if result.effect == Effect.APPLIED:
        if kind == OutcomeKind.SETTLEMENT:
            return f"Applied {result.applied_amount} against the outstanding balance."
        return "Obligation cancelled at the source; remaining balance withdrawn."
    if result.effect == Effect.EXCESS:
        return (
            f"{result.excess_amount} received beyond what was outstanding; recorded as "
            "excess, not counted as recovered."
        )
    return "No outstanding balance for this event to change."


def _apply_to_case(
    case,
    kind: str,
    result: ReconciliationResult,
    outcome_event_id: str,
    event_provenance: str | None = None,
) -> None:
    """Mutate `case` in place to reflect one applied SETTLEMENT or CANCELLATION event.
    Never called for PROMISE_CREATED / PROMISE_BROKEN, which never touch a case's
    financial position or state - see ingest_outcome_event."""
    ccy = case.amount_at_risk.currency
    case.recovered_amount = Money(amount=result.new_recovered, currency=ccy)
    case.outstanding_amount = Money(amount=result.new_outstanding, currency=ccy)
    case.cancelled_amount = Money(amount=result.new_cancelled, currency=ccy)
    try:
        CaseLifecycleManager.record_outcome_position(case, result.new_case_state)
    except CaseClosedError:
        # OUTCOME_FINAL_STATES already refuses this in reconcile_position (new_state ==
        # current_state whenever already_final), so this should never actually raise;
        # kept as a hard backstop rather than trusting that invariant silently.
        pass
    case.outcome = RecoveryOutcome(
        outcome_id=f"out_{uuid.uuid4().hex[:8]}",
        case_id=case.case_id,
        execution_id=case.execution_record.execution_id if case.execution_record else None,
        status=result.new_outcome_status,
        expected_amount=case.amount_at_risk,
        actual_amount_recovered=case.recovered_amount,
        verification_source="RECORDED_OUTCOME_EVENTS",
        external_reference=outcome_event_id,
        reconciliation_status=_reason_text(kind, result),
        verification_basis=domain_spec(case.risk_category).verification_basis,
        action_result=case.execution_record.status.value if case.execution_record else None,
        outstanding_amount=case.outstanding_amount,
        cancelled_amount=case.cancelled_amount,
        applied_outcome_event_ids=[
            *(case.outcome.applied_outcome_event_ids if case.outcome else []),
            outcome_event_id,
        ]
        if result.effect != Effect.NO_EFFECT
        else (case.outcome.applied_outcome_event_ids if case.outcome else []),
        provenance=_merged_provenance(
            case.outcome.provenance if case.outcome else None, event_provenance
        ),
    )


def _merged_provenance(previous: str | None, new: str | None) -> str | None:
    """The provenance to store after applying one more event: unchanged if this is
    the first, the same value if it agrees with what's already there, else "mixed" -
    never silently picks one side, since a case whose money came from two different
    kinds of evidence is not honestly described by either alone."""
    if previous is None:
        return new
    if new is None or new == previous:
        return previous
    return "mixed"


def ingest_outcome_event(db, case_repo, audit_repo, item: OutcomeEventInput) -> dict:
    """Validate, match, reconcile and persist one outcome event. Returns a JSON-safe
    result dict. Raises ValidationFailure for a malformed item (caller decides how to
    report it - see api/domain_router.py for the single and batch routes)."""
    fields = validate_outcome_item(item, db)
    outcome_event_id = f"oc_{uuid.uuid4().hex[:12]}"

    existing = (
        db.query(OutcomeEventModel)
        .filter(
            OutcomeEventModel.external_system == fields["external_system"],
            OutcomeEventModel.external_event_id == fields["external_event_id"],
        )
        .first()
    )
    if existing is not None:
        return _row_to_dict(existing, duplicate=True)

    resolved_customer_id, unresolved_reason = resolve_customer_id(
        db, fields["customer_id"], fields["customer_system"]
    )

    case = None
    match_reason = unresolved_reason
    if not match_reason:
        case, match_reason = find_matching_case(
            case_repo,
            domain=fields["domain"],
            customer_id=resolved_customer_id,
            reference_id=fields["reference_id"],
        )
    if (
        case is not None
        and fields["currency"] is not None
        and fields["currency"] != case.amount_at_risk.currency
    ):
        case, match_reason = None, MatchReason.CURRENCY_MISMATCH

    row_kwargs = dict(
        outcome_event_id=outcome_event_id,
        external_system=fields["external_system"],
        external_event_id=fields["external_event_id"],
        domain=fields["domain"].value,
        kind=fields["kind"],
        outcome_type=fields["outcome_type"],
        customer_id=fields["customer_id"],
        customer_system=fields["customer_system"],
        reference_id=fields["reference_id"],
        amount=fields["amount"],
        currency=fields["currency"],
        occurred_at=fields["occurred_at"].astimezone(UTC).replace(tzinfo=None),
        promised_date=fields["promised_date"].astimezone(UTC).replace(tzinfo=None)
        if fields["promised_date"]
        else None,
        raw_payload=item.raw_payload or {},
        source_type=item.source_type,
        source_ref=item.source_ref,
        ingestion_run_id=item.ingestion_run_id,
        provenance=fields["provenance"],
        evidence_reference=fields["evidence_reference"],
        source_id=fields["source_id"],
    )

    if case is None:
        row = OutcomeEventModel(
            **row_kwargs,
            match_status="UNMATCHED",
            match_reason=match_reason,
            case_id=None,
            applied_amount=None,
            excess_amount=None,
            effect=None,
        )
        db.add(row)
        audit_repo.log_transition(
            case_id="SYSTEM",
            from_state="N/A",
            to_state="N/A",
            evidence={
                "action": "outcome_unmatched",
                "outcome_event_id": outcome_event_id,
                "reason": match_reason,
                "domain": fields["domain"].value,
                "customer_id": fields["customer_id"],
                "reference_id": fields["reference_id"],
            },
        )
        db.commit()
        return _row_to_dict(row, duplicate=False)

    recovered, outstanding, cancelled = _decimals(case)
    result = reconcile_position(
        kind=fields["kind"],
        amount=fields["amount"],
        current_state=case.current_state,
        recovered=recovered,
        outstanding=outstanding,
        cancelled=cancelled,
    )
    before_state = case.current_state
    # Only a SETTLEMENT or CANCELLATION ever changes a case's financial position or
    # state; a promise is never itself money (reconcile_position already returns
    # NO_EFFECT for PROMISE_* with the case unchanged).
    money_kind = fields["kind"] in (OutcomeKind.SETTLEMENT, OutcomeKind.CANCELLATION)
    # And even a money-kind event only rewrites the case when it actually changed the
    # position (Effect.APPLIED). EXCESS (a settlement on an obligation that already has
    # nothing outstanding) and NO_EFFECT change nothing about the case: they are
    # recorded on the outcome_events row below for audit, but must NOT mint a new
    # case.outcome (a fresh outcome_id) over an unchanged position - that would make an
    # idempotent read (POST /cases/{id}/verify) look like a new determination every
    # time a duplicate or late settlement arrives.
    applied_to_case = money_kind and result.effect == Effect.APPLIED
    if applied_to_case:
        _apply_to_case(case, fields["kind"], result, outcome_event_id, fields["provenance"])

    row = OutcomeEventModel(
        **row_kwargs,
        match_status="MATCHED",
        match_reason=None,
        case_id=case.case_id,
        applied_amount=result.applied_amount,
        excess_amount=result.excess_amount,
        effect=result.effect,
    )
    db.add(row)

    if fields["kind"] == OutcomeKind.PROMISE_CREATED:
        db.add(
            PromiseModel(
                promise_id=f"prm_{uuid.uuid4().hex[:10]}",
                outcome_event_id=outcome_event_id,
                case_id=case.case_id,
                domain=fields["domain"].value,
                customer_id=fields["customer_id"],
                reference_id=fields["reference_id"],
                promised_amount=fields["amount"],
                currency=fields["currency"],
                promised_date=fields["promised_date"].astimezone(UTC).replace(tzinfo=None),
                created_at=row_kwargs["occurred_at"],
            )
        )
    elif fields["kind"] == OutcomeKind.PROMISE_BROKEN:
        open_promise = (
            db.query(PromiseModel)
            .filter(PromiseModel.case_id == case.case_id, PromiseModel.broken_reported_at.is_(None))
            .order_by(PromiseModel.created_at.desc())
            .first()
        )
        if open_promise is not None:
            open_promise.broken_reported_at = row_kwargs["occurred_at"]

    audit_repo.log_transition(
        case_id=case.case_id,
        from_state=before_state.value if hasattr(before_state, "value") else str(before_state),
        to_state=result.new_case_state.value
        if hasattr(result.new_case_state, "value")
        else str(result.new_case_state),
        evidence={
            "action": "outcome_reconciled" if money_kind else "outcome_recorded",
            "outcome_event_id": outcome_event_id,
            "kind": fields["kind"],
            "outcome_type": fields["outcome_type"],
            "effect": result.effect,
            "applied_amount": str(result.applied_amount),
            "excess_amount": str(result.excess_amount),
            "recovered_amount": str(result.new_recovered),
            "outstanding_amount": str(result.new_outstanding),
            "cancelled_amount": str(result.new_cancelled),
            "occurred_at": fields["occurred_at"].isoformat(),
        },
    )
    if applied_to_case:
        case_repo.save(case)
    else:
        db.flush()
    db.commit()
    return _row_to_dict(row, duplicate=False)


def rematch_unmatched_outcomes(db, case_repo, audit_repo, *, domain: str | None = None) -> dict:
    """Re-run matching for every stored UNMATCHED outcome event whose reason could be
    resolved by state that did not exist when it first arrived (an obligation created
    afterwards, or a customer-identity link recorded afterwards). This is the one place
    an outcome event's original decision is revisited, and only its MATCH - never an
    already-applied reconciliation. Explicit and auditable, never automatic: call it
    (e.g. after a batch of at-risk events, or a new identity link) rather than relying on
    it to run itself. Handles "settlement arrives before the failure event" safely
    without event-sourcing: the settlement is simply retried on request."""
    q = db.query(OutcomeEventModel).filter(
        OutcomeEventModel.match_status == "UNMATCHED",
        OutcomeEventModel.match_reason.in_(
            [MatchReason.NO_OBLIGATION, MatchReason.UNLINKED_CUSTOMER_IDENTITY]
        ),
    )
    if domain:
        q = q.filter(OutcomeEventModel.domain == domain)

    rematched = still_unmatched = 0
    for row in q.order_by(OutcomeEventModel.ingested_at).all():
        resolved_customer_id, unresolved_reason = resolve_customer_id(
            db, row.customer_id, row.customer_system
        )
        case = None
        if not unresolved_reason:
            case, _ = find_matching_case(
                case_repo,
                domain=RiskCategory(row.domain),
                customer_id=resolved_customer_id,
                reference_id=row.reference_id,
            )
        if row.currency and case is not None and row.currency != case.amount_at_risk.currency:
            case = None
        if case is None:
            still_unmatched += 1
            continue

        recovered, outstanding, cancelled = _decimals(case)
        result = reconcile_position(
            kind=row.kind,
            amount=row.amount,
            current_state=case.current_state,
            recovered=recovered,
            outstanding=outstanding,
            cancelled=cancelled,
        )
        before_state = case.current_state
        money_kind = row.kind in (OutcomeKind.SETTLEMENT, OutcomeKind.CANCELLATION)
        applied_to_case = money_kind and result.effect == Effect.APPLIED
        if applied_to_case:
            _apply_to_case(case, row.kind, result, row.outcome_event_id, row.provenance)
        elif row.kind == OutcomeKind.PROMISE_CREATED:
            db.add(
                PromiseModel(
                    promise_id=f"prm_{uuid.uuid4().hex[:10]}",
                    outcome_event_id=row.outcome_event_id,
                    case_id=case.case_id,
                    domain=row.domain,
                    customer_id=row.customer_id,
                    reference_id=row.reference_id,
                    promised_amount=row.amount,
                    currency=row.currency,
                    promised_date=row.promised_date,
                    created_at=row.occurred_at,
                )
            )
        row.match_status, row.match_reason, row.case_id = "MATCHED", None, case.case_id
        row.applied_amount, row.excess_amount, row.effect = (
            result.applied_amount,
            result.excess_amount,
            result.effect,
        )
        audit_repo.log_transition(
            case_id=case.case_id,
            from_state=before_state.value if hasattr(before_state, "value") else str(before_state),
            to_state=result.new_case_state.value
            if hasattr(result.new_case_state, "value")
            else str(result.new_case_state),
            evidence={
                "action": "outcome_rematched",
                "outcome_event_id": row.outcome_event_id,
                "effect": result.effect,
                "applied_amount": str(result.applied_amount),
            },
        )
        case_repo.save(case)
        rematched += 1
    db.commit()
    return {"rematched": rematched, "still_unmatched": still_unmatched}


def _row_to_dict(row: OutcomeEventModel, duplicate: bool) -> dict:
    return {
        "outcome_event_id": row.outcome_event_id,
        "status": "duplicate" if duplicate else "processed",
        "external_system": row.external_system,
        "external_event_id": row.external_event_id,
        "domain": row.domain,
        "kind": row.kind,
        "outcome_type": row.outcome_type,
        "customer_id": row.customer_id,
        "reference_id": row.reference_id,
        "amount": str(row.amount) if row.amount is not None else None,
        "currency": row.currency,
        "occurred_at": row.occurred_at.isoformat() if row.occurred_at else None,
        "match_status": row.match_status,
        "match_reason": row.match_reason,
        "case_id": row.case_id,
        "applied_amount": str(row.applied_amount) if row.applied_amount is not None else None,
        "excess_amount": str(row.excess_amount) if row.excess_amount is not None else None,
        "effect": row.effect,
        "provenance": row.provenance,
        "evidence_reference": row.evidence_reference,
        "source_id": row.source_id,
    }


def derive_promise_status(
    promise: PromiseModel, applied_settlement_total: Decimal, as_of: datetime
) -> str:
    """CREATED / DUE / FULFILLED / BROKEN, derived deterministically - never stored
    (except an explicit PROMISE_BROKEN report). ``applied_settlement_total`` is every
    SETTLEMENT amount applied to the case between the promise's creation and the end of
    its promised day (UTC); a promise is never itself money. ``end of promised_date`` is
    23:59:59 UTC on that calendar day - promised_date carries no time-of-day, and UTC is
    the same convention application/ingestion.py uses for every other naive timestamp."""
    promised_day_end = promise.promised_date.replace(hour=23, minute=59, second=59, microsecond=0)
    if applied_settlement_total >= promise.promised_amount:
        return "FULFILLED"
    if promise.broken_reported_at is not None:
        return "BROKEN"
    if as_of > promised_day_end:
        return "BROKEN"
    if as_of >= promise.promised_date.replace(hour=0, minute=0, second=0, microsecond=0):
        return "DUE"
    return "CREATED"
