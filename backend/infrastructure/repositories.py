from application.case_engine import ACTIVE_STATES
from domain.interfaces import ICaseRepository
from domain.models import (
    ActionRecommendation,
    ExecutionRecord,
    PolicyDecision,
    RecoveryCase,
    RecoveryOutcome,
    RecoveryPrediction,
    RevenueEvent,
    RiskAssessment,
    RootCauseDiagnosis,
)
from domain.revenue_domains import obligation_key


class InMemoryCaseRepository(ICaseRepository):
    """A purely in-memory repository for robust unit testing without database constraints."""

    def __init__(self):
        self.cases: dict[str, RecoveryCase] = {}
        self.events: dict[str, RevenueEvent] = {}

    def save(self, case: RecoveryCase) -> None:
        self.cases[case.case_id] = case
        for event in case.linked_events:
            self.events[(event.external_system, event.external_event_id)] = event

    def get_by_id(self, case_id: str) -> RecoveryCase | None:
        return self.cases.get(case_id)

    def get_event_by_external_id(
        self, external_system: str, external_event_id: str
    ) -> RevenueEvent | None:
        return self.events.get((external_system, external_event_id))

    def get_active_case_for_customer(
        self, customer_id: str, risk_category: str, reference_id: str | None
    ) -> RecoveryCase | None:
        wanted = obligation_key(customer_id, risk_category, reference_id)
        if wanted is None:
            return None
        for case in self.cases.values():
            if obligation_key(case.customer_id, case.risk_category, case.reference_id) == wanted:
                if case.current_state in ACTIVE_STATES:
                    return case
        return None


import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from domain.interfaces import IAuditRecorder
from domain.models import Money
from infrastructure.orm import (
    AuditModel,
    CaseModel,
    ContactPermissionModel,
    EventIdentityAliasModel,
    EventModel,
    ExecutionAttemptModel,
    IdempotencyRecord,
)


def _attempt_channel(action_type: str) -> str:
    """Channel an attempt is recorded under. Delegates to the single domain-level
    classification so the ledger and the Policy Engine's rate limits can never
    disagree about what counts as customer contact."""
    from domain.models import action_channel

    return action_channel(action_type).value


def _event_provenance(e) -> dict:
    """Provenance fields of an EventModel row, for rehydrating a RevenueEvent.
    ingested_at is passed only when it was actually recorded (pre-Phase-3 rows have
    none), so a legacy event is never given a made-up ingestion time."""
    out = {
        "source_type": e.source_type,
        "source_ref": e.source_ref,
        "ingestion_run_id": e.ingestion_run_id,
        "domain_basis": e.domain_basis,
    }
    if e.ingested_at is not None:
        out["ingested_at"] = (
            e.ingested_at if e.ingested_at.tzinfo else e.ingested_at.replace(tzinfo=UTC)
        )
    return out


class SqlAlchemyAuditRecorder(IAuditRecorder):
    def __init__(self, db: Session):
        self.db = db

    def has_idempotency_key(self, key: str) -> bool:
        return (
            self.db.query(IdempotencyRecord)
            .filter(IdempotencyRecord.idempotency_key == key)
            .first()
            is not None
        )

    def get_execution_record_by_key(self, key: str) -> dict:
        record = (
            self.db.query(IdempotencyRecord)
            .filter(IdempotencyRecord.idempotency_key == key)
            .first()
        )
        return record.execution_record if record else None

    def save_idempotency_key(self, key: str, execution_record: dict) -> None:
        rec = IdempotencyRecord(idempotency_key=key, execution_record=execution_record)
        self.db.add(rec)
        self.db.flush()

    def log_transition(self, case_id: str, from_state: str, to_state: str, evidence: dict) -> None:
        # Phase 16: every case-scoped audit record carries the journey trace_id. Derived
        # from the case id, so all stages agree without coordination; a stage that
        # already supplied one (it must be the same value) is left untouched.
        if case_id and case_id != "SYSTEM" and isinstance(evidence, dict):
            from application.trace_ids import trace_id_for_case

            evidence = {**evidence}
            evidence.setdefault("trace_id", trace_id_for_case(case_id))
        audit = AuditModel(
            id=uuid.uuid4().hex,
            case_id=case_id if case_id != "SYSTEM" else None,
            from_state=from_state,
            to_state=to_state,
            evidence=evidence,
        )
        self.db.add(audit)
        # Flush to ensure it is in the transaction, but don't commit yet (Repository commits)
        self.db.flush()


class SqlAlchemyCaseRepository(ICaseRepository):
    def __init__(self, db: Session):
        self.db = db

    def get_by_id_for_update(self, case_id: str):
        c = self.db.query(CaseModel).with_for_update().filter(CaseModel.case_id == case_id).first()
        if not c:
            return None
        return self.get_by_id(case_id)

    def has_idempotency_key(self, key: str) -> bool:
        return (
            self.db.query(IdempotencyRecord)
            .filter(IdempotencyRecord.idempotency_key == key)
            .first()
            is not None
        )

    def get_execution_record_by_key(self, key: str) -> dict:
        record = (
            self.db.query(IdempotencyRecord)
            .filter(IdempotencyRecord.idempotency_key == key)
            .first()
        )
        return record.execution_record if record else None

    def save_idempotency_key(self, key: str, execution_record: dict) -> None:
        rec = IdempotencyRecord(idempotency_key=key, execution_record=execution_record)
        self.db.add(rec)
        self.db.flush()

    # -- execution slot claim (concurrency control) -------------------------
    #
    # An execution must happen at most once per authorization, even if two requests
    # arrive at the same instant. The claim below is an INSERT of the idempotency key,
    # which is the table's PRIMARY KEY -- so the database itself, not application
    # sequencing, decides who wins. This is the one guarantee that holds identically on
    # SQLite and PostgreSQL, which is why it is used instead of relying on
    # with_for_update() (a no-op under SQLite).

    _CLAIM_PLACEHOLDER = {"status": "CLAIMED", "note": "execution in flight"}

    def claim_execution_slot(self, key: str) -> bool:
        """Try to reserve the right to execute `key`. True if this caller won the
        claim; False if it was already taken (duplicate / concurrent request).

        On conflict the session is rolled back: this is called BEFORE any of the
        request's own writes, so nothing of value is discarded.
        """
        from sqlalchemy.exc import IntegrityError

        self.db.add(
            IdempotencyRecord(idempotency_key=key, execution_record=self._CLAIM_PLACEHOLDER)
        )
        try:
            self.db.flush()
            return True
        except IntegrityError:
            self.db.rollback()
            return False

    def complete_execution_slot(self, key: str, execution_record: dict) -> None:
        """Write the real result over the placeholder left by claim_execution_slot."""
        from sqlalchemy.orm.attributes import flag_modified

        rec = (
            self.db.query(IdempotencyRecord)
            .filter(IdempotencyRecord.idempotency_key == key)
            .first()
        )
        if rec is None:  # claim lost to a rollback elsewhere - recreate it
            self.db.add(IdempotencyRecord(idempotency_key=key, execution_record=execution_record))
        else:
            rec.execution_record = execution_record
            flag_modified(rec, "execution_record")
        self.db.flush()

    @staticmethod
    def is_claim_placeholder(execution_record: dict) -> bool:
        return bool(execution_record) and execution_record.get("status") == "CLAIMED"

    # -- execution attempt ledger ------------------------------------------
    def record_execution_attempt(
        self,
        case_id: str,
        action_type: str,
        status: str,
        idempotency_key: str = None,
        execution_id: str = None,
        agent_type: str = None,
        policy_decision_id: str = None,
        contact_permission: str = None,
        trace_id: str = None,
    ) -> None:
        self.db.add(
            ExecutionAttemptModel(
                id=uuid.uuid4().hex,
                case_id=case_id,
                action_type=action_type,
                channel=_attempt_channel(action_type),
                status=status,
                idempotency_key=idempotency_key,
                execution_id=execution_id,
                agent_type=agent_type,
                policy_decision_id=policy_decision_id,
                contact_permission=contact_permission,
                trace_id=trace_id,
            )
        )
        self.db.flush()

    def get_policy_context(self, case_id: str, approved_decision_id: str | None = None):
        """Build a PolicyContext from durable state.

        Sources: the execution_attempts ledger (retry / contact history), the case's
        own lifecycle state (the STOPPED hard stop), and the append-only audit trail
        (whether a human approved a specific policy decision).

        `approved_decision_id` must be supplied ONLY when revalidating the decision a
        human actually reviewed - i.e. immediately before executing under it. A fresh
        policy evaluation deliberately passes nothing, so a new decision never
        inherits an earlier approval.
        """
        from application.policy_engine import PolicyContext
        from domain.models import CaseState

        now = datetime.now(UTC)

        # Scope history to the OBLIGATION, not just this case row.
        #
        # Correlation refuses to attach a new event to a closed case, so the next
        # failure for the same customer + risk_category + reference_id opens a brand
        # new case. Counting attempts by case_id alone therefore handed that new case
        # a fresh retry budget and an empty cooldown window, and forgot that the
        # customer had opted out - the retry cap, the 24h contact cap and the STOP
        # would all silently reset. Sibling cases for the same obligation are counted
        # together. When there is no reference_id the obligation cannot be identified,
        # so we fall back to this case alone - the same conservative rule
        # case_engine.py already applies when deciding whether to correlate at all.
        db_case = self.db.query(CaseModel).filter(CaseModel.case_id == case_id).first()
        sibling_ids = [case_id]
        sibling_states = []
        if db_case is not None:
            sibling_states = [db_case.current_state]
            # Siblings share the obligation identity (domain.revenue_domains.obligation_key).
            if obligation_key(db_case.customer_id, db_case.risk_category, db_case.reference_id):
                siblings = (
                    self.db.query(CaseModel.case_id, CaseModel.current_state)
                    .filter(
                        CaseModel.customer_id == db_case.customer_id,
                        CaseModel.risk_category == db_case.risk_category,
                        CaseModel.reference_id == db_case.reference_id,
                    )
                    .all()
                )
                if siblings:
                    sibling_ids = [s[0] for s in siblings]
                    sibling_states = [s[1] for s in siblings]

        rows = (
            self.db.query(ExecutionAttemptModel)
            .filter(ExecutionAttemptModel.case_id.in_(sibling_ids))
            .all()
        )
        pay = [r for r in rows if r.channel == "payment"]
        comms = [r for r in rows if r.channel == "communication"]
        last_pay = max((r.created_at for r in pay), default=None)
        window = now - timedelta(hours=24)

        # Contact frequency limits how often a PERSON is contacted, so communication
        # attempts are counted across every case of this customer. Counting them per
        # obligation let a customer with three overdue invoices receive the 24h cap once
        # per invoice. (Payment retries stay per obligation: each obligation has its own
        # retry budget.) This is a superset of the obligation's own history, so the cap
        # can only become stricter.
        customer_id = db_case.customer_id if db_case is not None else None
        if customer_id is not None:
            comms = (
                self.db.query(ExecutionAttemptModel)
                .join(CaseModel, CaseModel.case_id == ExecutionAttemptModel.case_id)
                .filter(
                    CaseModel.customer_id == customer_id,
                    ExecutionAttemptModel.channel == "communication",
                )
                .all()
            )
        comms_24h = sum(
            1
            for r in comms
            if r.created_at
            and (r.created_at if r.created_at.tzinfo else r.created_at.replace(tzinfo=UTC))
            >= window
        )

        # Contact permission from the append-only contact_permissions table. No record
        # resolves to UNKNOWN, which the Policy Engine never treats as consent.
        from application.contact_permissions import resolve_all

        permission_rows = (
            self.db.query(ContactPermissionModel)
            .filter(ContactPermissionModel.customer_id == customer_id)
            .all()
            if customer_id is not None
            else []
        )
        permissions = resolve_all(permission_rows, at=now)
        # Hard stop: honour a STOP recorded against ANY case for this obligation.
        # Previously nothing ever set this field, so PolicyEngine's StopRule was dead
        # on every live path; scoping it to the obligation also means a customer who
        # opted out is not contacted again via a freshly created case.
        stopped = any(s == CaseState.STOPPED for s in sibling_states)

        # Human approval, scoped to the exact policy decision that was reviewed.
        human_approved = False
        if approved_decision_id:
            for a in (
                self.db.query(AuditModel)
                .filter(AuditModel.case_id == case_id)
                .order_by(AuditModel.timestamp.desc())
                .limit(200)
                .all()
            ):
                ev = a.evidence or {}
                if (
                    ev.get("action") == "human_review"
                    and str(ev.get("decision", "")).upper() == "APPROVE"
                    and ev.get("approved_decision_id") == approved_decision_id
                ):
                    human_approved = True
                    break

        return PolicyContext(
            prior_payment_attempts=len(pay),
            last_payment_attempt_at=last_pay,
            comms_in_last_24h=comms_24h,
            stopped=stopped,
            communication_permission=permissions["communication"].status,
            payment_permission=permissions["payment"].status,
            permission_records={
                channel: resolved.to_dict()
                for channel, resolved in permissions.items()
                if resolved.record_id
            },
            human_approved=human_approved,
        )

    def save(self, case: RecoveryCase) -> None:
        # Check if case exists
        db_case = self.db.query(CaseModel).filter(CaseModel.case_id == case.case_id).first()
        if not db_case:
            db_case = CaseModel(
                case_id=case.case_id,
                customer_id=case.customer_id,
                risk_category=case.risk_category,
                reference_id=case.reference_id,
                amount_at_risk=case.amount_at_risk.amount,
                expected_recoverable_value=case.expected_recoverable_value.amount
                if case.expected_recoverable_value
                else None,
                currency=case.amount_at_risk.currency,
                current_state=case.current_state,
                risk_assessment=case.risk_assessment.model_dump(mode="json")
                if case.risk_assessment
                else None,
                diagnosis=case.diagnosis.model_dump(mode="json") if case.diagnosis else None,
                prediction=case.prediction.model_dump(mode="json") if case.prediction else None,
                recommendation=case.recommendation.model_dump(mode="json")
                if case.recommendation
                else None,
                policy_decision=case.policy_decision.model_dump(mode="json")
                if case.policy_decision
                else None,
                execution_record=case.execution_record.model_dump(mode="json")
                if case.execution_record
                else None,
                outcome=case.outcome.model_dump(mode="json") if case.outcome else None,
                recovered_amount=case.recovered_amount.amount if case.recovered_amount else 0,
                outstanding_amount=case.outstanding_amount.amount
                if case.outstanding_amount is not None
                else case.amount_at_risk.amount,
                cancelled_amount=case.cancelled_amount.amount if case.cancelled_amount else 0,
            )
            self.db.add(db_case)
        else:
            from sqlalchemy.orm.attributes import flag_modified

            db_case.amount_at_risk = case.amount_at_risk.amount
            db_case.expected_recoverable_value = (
                case.expected_recoverable_value.amount if case.expected_recoverable_value else None
            )
            db_case.current_state = case.current_state
            # Phase 5 financial position (application/outcome_events.py). Written
            # unconditionally - unlike the JSON blobs below, 0 is a real, meaningful
            # value here (a Decimal 0 is falsy in Python, so `if case.recovered_amount`
            # would silently skip persisting a reconciliation that zeroed a field out).
            if case.recovered_amount is not None:
                db_case.recovered_amount = case.recovered_amount.amount
            if case.outstanding_amount is not None:
                db_case.outstanding_amount = case.outstanding_amount.amount
            if case.cancelled_amount is not None:
                db_case.cancelled_amount = case.cancelled_amount.amount
            if case.risk_assessment:
                db_case.risk_assessment = case.risk_assessment.model_dump(mode="json")
                flag_modified(db_case, "risk_assessment")
            if case.diagnosis:
                db_case.diagnosis = case.diagnosis.model_dump(mode="json")
                flag_modified(db_case, "diagnosis")
            if case.prediction:
                db_case.prediction = case.prediction.model_dump(mode="json")
                flag_modified(db_case, "prediction")
            if case.recommendation:
                db_case.recommendation = case.recommendation.model_dump(mode="json")
                flag_modified(db_case, "recommendation")
            if case.policy_decision:
                db_case.policy_decision = case.policy_decision.model_dump(mode="json")
                flag_modified(db_case, "policy_decision")
            if case.execution_record:
                db_case.execution_record = case.execution_record.model_dump(mode="json")
                flag_modified(db_case, "execution_record")
            if hasattr(case, "outcome") and case.outcome:
                db_case.outcome = case.outcome.model_dump(mode="json")
                flag_modified(db_case, "outcome")

        # Sync events
        existing_event_ids = {e.event_id for e in db_case.events} if db_case.events else set()
        for event in case.linked_events:
            if event.event_id not in existing_event_ids:
                db_event = EventModel(
                    event_id=event.event_id,
                    case_id=case.case_id,
                    customer_id=event.customer_id,
                    external_system=event.external_system,
                    external_event_id=event.external_event_id,
                    reference_id=event.reference_id,
                    risk_category=event.risk_category,
                    amount=event.amount.amount,
                    currency=event.amount.currency,
                    timestamp=event.timestamp,
                    raw_payload=event.raw_payload,
                    source_type=event.source_type,
                    source_ref=event.source_ref,
                    ingestion_run_id=event.ingestion_run_id,
                    ingested_at=event.ingested_at,
                    domain_basis=event.domain_basis,
                )
                self.db.add(db_event)

        # We rely on the caller/dependency injector to call db.commit() OR we commit here.
        # To make it transactional at the use-case level, we commit here — unless a
        # bulk caller (e.g. /datasets/{id}/generate-cases) set `_defer_commit`, in
        # which case we only flush so in-loop reads still see the row and the whole
        # batch is committed once at the end.
        if getattr(self, "_defer_commit", False):
            self.db.flush()
        else:
            self.db.commit()

    def get_by_id(self, case_id: str) -> RecoveryCase | None:
        # Skipping full mapping for brevity in this foundational phase,
        # normally you would map the SQLAlchemy model back to the Pydantic Domain Model.
        # But we need it for the GET API.
        db_case = self.db.query(CaseModel).filter(CaseModel.case_id == case_id).first()
        if not db_case:
            return None

        case = RecoveryCase(
            case_id=db_case.case_id,
            customer_id=db_case.customer_id,
            risk_category=db_case.risk_category,
            reference_id=db_case.reference_id,
            amount_at_risk=Money(amount=db_case.amount_at_risk, currency=db_case.currency),
            expected_recoverable_value=Money(
                amount=db_case.expected_recoverable_value, currency=db_case.currency
            )
            if db_case.expected_recoverable_value is not None
            else None,
            current_state=db_case.current_state,
            risk_assessment=RiskAssessment(**db_case.risk_assessment)
            if db_case.risk_assessment
            else None,
            diagnosis=RootCauseDiagnosis(**db_case.diagnosis) if db_case.diagnosis else None,
            prediction=RecoveryPrediction(**db_case.prediction) if db_case.prediction else None,
            recommendation=ActionRecommendation(**db_case.recommendation)
            if db_case.recommendation
            else None,
            policy_decision=PolicyDecision(**db_case.policy_decision)
            if db_case.policy_decision
            else None,
            execution_record=ExecutionRecord(**db_case.execution_record)
            if db_case.execution_record
            else None,
            outcome=RecoveryOutcome(**db_case.outcome) if db_case.outcome else None,
            recovered_amount=Money(amount=db_case.recovered_amount, currency=db_case.currency),
            outstanding_amount=Money(
                amount=db_case.outstanding_amount
                if db_case.outstanding_amount is not None
                else db_case.amount_at_risk,
                currency=db_case.currency,
            ),
            cancelled_amount=Money(amount=db_case.cancelled_amount, currency=db_case.currency),
            linked_events=[
                RevenueEvent(
                    event_id=e.event_id,
                    customer_id=e.customer_id,
                    external_system=e.external_system,
                    external_event_id=e.external_event_id,
                    reference_id=e.reference_id,
                    risk_category=e.risk_category,
                    amount=Money(amount=e.amount, currency=e.currency),
                    timestamp=e.timestamp.replace(tzinfo=UTC)
                    if e.timestamp.tzinfo is None
                    else e.timestamp,
                    raw_payload=e.raw_payload,
                    **_event_provenance(e),
                )
                for e in db_case.events
            ],
        )
        return case

    def find_existing_event(self, external_system: str, external_event_ids: list[str]):
        """The stored event recognised under ANY of these ids, as its own external id or
        as an identity alias (event_identity_aliases), or None. Ids are tried in the
        order given, so the current identity wins over legacy ones."""
        ids = list(dict.fromkeys(i for i in external_event_ids if i))
        if not ids:
            return None
        direct = {
            e.external_event_id: e
            for e in self.db.query(EventModel)
            .filter(
                EventModel.external_system == external_system,
                EventModel.external_event_id.in_(ids),
            )
            .all()
        }
        aliased = {
            a.alias_event_id: a.event_id
            for a in self.db.query(EventIdentityAliasModel)
            .filter(
                EventIdentityAliasModel.alias_system == external_system,
                EventIdentityAliasModel.alias_event_id.in_(ids),
            )
            .all()
        }
        for i in ids:
            if i in direct:
                return direct[i]
            if i in aliased:
                return self.db.query(EventModel).filter(EventModel.event_id == aliased[i]).first()
        return None

    def get_event_by_external_id(
        self, external_system: str, external_event_id: str
    ) -> RevenueEvent | None:
        # Deduplication honours identity aliases written by identity reconciliation, so
        # an event stored under a legacy identity is recognised under its current one.
        e = self.find_existing_event(external_system, [external_event_id])
        if e:
            return RevenueEvent(
                event_id=e.event_id,
                customer_id=e.customer_id,
                external_system=e.external_system,
                external_event_id=e.external_event_id,
                reference_id=e.reference_id,
                risk_category=e.risk_category,
                amount=Money(amount=e.amount, currency=e.currency),
                timestamp=e.timestamp.replace(tzinfo=UTC)
                if e.timestamp.tzinfo is None
                else e.timestamp,
                raw_payload=e.raw_payload,
                **_event_provenance(e),
            )
        return None

    def get_active_case_for_customer(
        self, customer_id: str, risk_category: str, reference_id: str | None
    ) -> RecoveryCase | None:
        if not reference_id:
            return None

        db_case = (
            self.db.query(CaseModel)
            .filter(
                CaseModel.customer_id == customer_id,
                CaseModel.risk_category == risk_category,
                CaseModel.reference_id == reference_id,
                CaseModel.current_state.in_(ACTIVE_STATES),
            )
            .first()
        )

        if db_case:
            return self.get_by_id(db_case.case_id)
        return None

    def get_cases_for_obligation(
        self, customer_id: str, risk_category: str, reference_id: str
    ) -> list[RecoveryCase]:
        """Every case (any state) sharing this obligation (domain.revenue_domains.
        obligation_key), oldest first. Unlike get_active_case_for_customer, this is used
        for OUTCOME matching (application/outcome_events.py): an outcome event must be
        able to reach a case regardless of its current state - a settlement can arrive
        for an obligation whose case already closed, and closed cases besides the most
        recent are still real financial history that late money can apply to."""
        if not reference_id:
            return []
        rows = (
            self.db.query(CaseModel)
            .filter(
                CaseModel.customer_id == customer_id,
                CaseModel.risk_category == risk_category,
                CaseModel.reference_id == reference_id,
            )
            .order_by(CaseModel.created_at)
            .all()
        )
        return [self.get_by_id(r.case_id) for r in rows]


def _naive_utc(dt: datetime) -> datetime:
    """Stored DateTime columns hold naive UTC (SQLite drops an offset without converting)."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(UTC).replace(tzinfo=None)


class SqlAlchemyContactPermissionRepository:
    """Append-only store of contact permissions (application/contact_permissions.py)."""

    def __init__(self, db: Session):
        self.db = db

    def record(
        self,
        *,
        customer_id: str,
        channel: str,
        status: str,
        source: str,
        reason: str | None = None,
        effective_at: datetime | None = None,
        source_record_id: str | None = None,
        ingestion_run_id: str | None = None,
    ) -> tuple[ContactPermissionModel, bool]:
        """(row, is_new). A byte-identical record for this customer (same channel,
        status, source, reason, effective_at - see application.contact_permissions.
        content_hash) is recognised rather than stored again, so re-running a consent
        import is safe. A genuine repeated change of mind (opt out, allow, opt out
        again) always has a different effective_at and is stored as its own row."""
        from application.contact_permissions import content_hash as _content_hash
        from application.contact_permissions import validate_permission

        validate_permission(channel, status)
        now = datetime.now(UTC)
        effective_at = effective_at or now
        digest = _content_hash(
            customer_id=customer_id,
            channel=channel,
            status=status,
            source=source,
            reason=reason,
            effective_at=effective_at,
        )
        existing = (
            self.db.query(ContactPermissionModel)
            .filter(
                ContactPermissionModel.customer_id == customer_id,
                ContactPermissionModel.content_hash == digest,
            )
            .first()
        )
        if existing is not None:
            return existing, False
        row = ContactPermissionModel(
            id=f"perm_{uuid.uuid4().hex[:12]}",
            customer_id=customer_id,
            channel=channel,
            status=status,
            source=source,
            reason=reason,
            effective_at=_naive_utc(effective_at),
            recorded_at=_naive_utc(now),
            source_record_id=source_record_id,
            ingestion_run_id=ingestion_run_id,
            content_hash=digest,
        )
        self.db.add(row)
        self.db.flush()
        return row, True

    def list_for_customer(self, customer_id: str) -> list[ContactPermissionModel]:
        return (
            self.db.query(ContactPermissionModel)
            .filter(ContactPermissionModel.customer_id == customer_id)
            .order_by(ContactPermissionModel.effective_at, ContactPermissionModel.recorded_at)
            .all()
        )
