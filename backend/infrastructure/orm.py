from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from domain.models import CaseState, RiskCategory
from infrastructure.db import Base


class CaseModel(Base):
    __tablename__ = "recovery_cases"

    case_id = Column(String, primary_key=True, index=True)
    customer_id = Column(String, index=True, nullable=False)
    risk_category = Column(Enum(RiskCategory), nullable=False)
    reference_id = Column(String, index=True, nullable=True)

    amount_at_risk = Column(Numeric(precision=18, scale=4), nullable=False)
    expected_recoverable_value = Column(Numeric(precision=18, scale=4), nullable=True)
    currency = Column(String, default="INR", nullable=False)

    current_state = Column(Enum(CaseState), nullable=False, default=CaseState.DETECTED)
    risk_assessment = Column(JSON, nullable=True)
    diagnosis = Column(JSON, nullable=True)
    prediction = Column(JSON, nullable=True)
    recommendation = Column(JSON, nullable=True)
    policy_decision = Column(JSON, nullable=True)
    execution_record = Column(JSON, nullable=True)
    outcome = Column(JSON, nullable=True)

    # Phase 5: the case's current financial position, maintained ONLY by
    # application.outcome_events.reconcile_case from the append-only outcome_events
    # table (never by reprocessing risk/diagnosis/policy). recovered_amount +
    # outstanding_amount + cancelled_amount == amount_at_risk at the time each was
    # applied (see that module's docstring for the one documented exception:
    # settlement received after the obligation was already cancelled or fully
    # recovered, which is applied as excess and audited, never added here).
    recovered_amount = Column(
        Numeric(precision=18, scale=4), nullable=False, default=0, server_default="0"
    )
    # Nullable: rows inserted directly (bypassing SqlAlchemyCaseRepository.save, e.g. in
    # some tests) may not set it. NULL is read back as "not yet computed" and treated as
    # the full amount_at_risk still outstanding (infrastructure/repositories.py
    # get_by_id), so it is never ambiguous with a real, reconciled zero.
    outstanding_amount = Column(Numeric(precision=18, scale=4), nullable=True)
    cancelled_amount = Column(
        Numeric(precision=18, scale=4), nullable=False, default=0, server_default="0"
    )

    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    updated_at = Column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    # Optimistic concurrency. SQLAlchemy appends "WHERE version_id = :read_version" to
    # every UPDATE of this row and raises StaleDataError when no row matches, i.e.
    # when someone else changed the case since we read it. This is the protection that
    # with_for_update() cannot provide on SQLite (the dialect drops FOR UPDATE
    # entirely), and it behaves identically on PostgreSQL. Without it the repository's
    # read-modify-write is last-writer-wins, so a stage request that read a case just
    # before verification committed could write POLICY_EVALUATED back over
    # FULLY_RECOVERED and make a settled case look reprocessable again.
    version_id = Column(Integer, nullable=False, default=1, server_default="1")

    __mapper_args__ = {"version_id_col": version_id}

    events = relationship("EventModel", back_populates="case")
    audits = relationship("AuditModel", back_populates="case")


class EventModel(Base):
    __tablename__ = "revenue_events"

    event_id = Column(String, primary_key=True, index=True)
    case_id = Column(String, ForeignKey("recovery_cases.case_id"), nullable=False)
    customer_id = Column(String, nullable=False)
    external_system = Column(String, nullable=False)
    external_event_id = Column(String, nullable=False, index=True)
    reference_id = Column(String, index=True, nullable=True)
    risk_category = Column(Enum(RiskCategory), nullable=False)

    amount = Column(Numeric(precision=18, scale=4), nullable=False)
    currency = Column(String, default="INR")

    timestamp = Column(DateTime, nullable=False)
    raw_payload = Column(JSON, nullable=False)

    # Provenance (Phase 3): where this event entered the system and under which run.
    # Before this, a file-replayed event recorded only "Dataset-<id>" in
    # external_system and RevenueEvent.ingested_at was never persisted. Nullable:
    # events ingested before Phase 3 simply carry no provenance.
    source_type = Column(String, nullable=True)  # application.ingestion.SourceType
    source_ref = Column(String, index=True, nullable=True)  # dataset_id for file replay
    ingestion_run_id = Column(String, index=True, nullable=True)
    ingested_at = Column(DateTime, nullable=True)

    # Why this event carries its risk_category (domain.revenue_domains.DomainBasis):
    # declared by the source, declared for the dataset, a category column, or named
    # evidence. Nullable: events stored before Phase 4 never recorded it.
    domain_basis = Column(String, nullable=True)

    case = relationship("CaseModel", back_populates="events")

    __table_args__ = (
        UniqueConstraint("external_system", "external_event_id", name="uq_external_event_sys_id"),
    )


class AuditModel(Base):
    __tablename__ = "audit_records"

    id = Column(String, primary_key=True)
    case_id = Column(
        String, ForeignKey("recovery_cases.case_id"), index=True, nullable=True
    )  # Nullable for system-level audits like dedup
    from_state = Column(String)
    to_state = Column(String)
    evidence = Column(JSON)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))

    case = relationship("CaseModel", back_populates="audits")


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_keys"

    idempotency_key = Column(String, primary_key=True)
    execution_record = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))


class ExecutionAttemptModel(Base):
    """Append-only ledger of every recovery action the system has attempted for a
    case. Used by the Policy Engine to enforce retry / cooldown / contact limits
    against real history instead of a proxy on event counts."""

    __tablename__ = "execution_attempts"

    id = Column(String, primary_key=True)
    case_id = Column(String, ForeignKey("recovery_cases.case_id"), index=True, nullable=False)
    action_type = Column(String, nullable=False)
    channel = Column(String, nullable=True)  # "payment" | "communication" | "internal"
    status = Column(String, nullable=False)  # ExecutionStatus value
    idempotency_key = Column(String, index=True, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), index=True)

    # Added in Phase 2 so the ledger alone can answer "which agent did what, when,
    # and under whose authorization". CaseModel.execution_record holds only the LAST
    # execution, so a case with several attempts previously reported just the most
    # recent one. Nullable: rows written before this change simply have no value.
    execution_id = Column(String, index=True, nullable=True)
    agent_type = Column(String, nullable=True)

    # Added in Phase 4 so contact history answers "under which policy decision?" and
    # "what was the customer's permission at that moment?" from the ledger itself.
    # contact_permission is the status the Policy Engine evaluated for the action's
    # channel (ALLOWED / UNKNOWN; OPTED_OUT and SUPPRESSED never reach execution).
    policy_decision_id = Column(String, index=True, nullable=True)
    contact_permission = Column(String, nullable=True)

    # Phase 16: the journey this attempt belongs to (application/trace_ids.py).
    trace_id = Column(String, index=True, nullable=True)


class IngestionRunModel(Base):
    """One batch ingestion (a dataset replay or an /events/batch call) and exactly
    what happened to its records. Written once, in the run's own transaction, so the
    data-quality result outlives the HTTP response that used to be its only record."""

    __tablename__ = "ingestion_runs"

    run_id = Column(String, primary_key=True)
    source_type = Column(String, nullable=False)  # application.ingestion.SourceType
    source_ref = Column(String, index=True, nullable=True)  # dataset_id for file replay
    source_name = Column(String, nullable=True)  # original filename, when there is one
    status = Column(String, nullable=False)  # COMPLETED | FAILED
    started_at = Column(DateTime, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)

    rows_in_source = Column(Integer, nullable=True)  # rows the source holds in total
    rows_received = Column(Integer, nullable=False, default=0, server_default="0")
    rows_accepted = Column(Integer, nullable=False, default=0, server_default="0")
    rows_rejected = Column(Integer, nullable=False, default=0, server_default="0")
    rows_duplicate = Column(Integer, nullable=False, default=0, server_default="0")
    rows_not_at_risk = Column(Integer, nullable=False, default=0, server_default="0")
    rows_not_processed = Column(Integer, nullable=False, default=0, server_default="0")
    cases_created = Column(Integer, nullable=False, default=0, server_default="0")
    cases_updated = Column(Integer, nullable=False, default=0, server_default="0")

    rejection_counts = Column(JSON, nullable=True)  # {reason_code: count}
    rejected_samples = Column(JSON, nullable=True)  # bounded [{row, reason, field, detail}]
    # {classification: <how rows got their domain>, accepted_by_domain: {domain: n}}
    domain_classification = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)


class ContactPermissionModel(Base):
    """What a customer has permitted, per channel. Append-only: a change of mind is a
    new row, so the permission in force at any past moment can be reconstructed and the
    history is never rewritten. Resolution: application/contact_permissions.py."""

    __tablename__ = "contact_permissions"

    id = Column(String, primary_key=True)
    customer_id = Column(String, index=True, nullable=False)
    channel = Column(String, nullable=False)  # communication | payment | all
    status = Column(String, nullable=False)  # ALLOWED | OPTED_OUT | SUPPRESSED
    source = Column(String, nullable=False)  # who/what recorded it, e.g. customer_request
    reason = Column(String, nullable=True)
    effective_at = Column(DateTime, nullable=False, index=True)
    recorded_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    # Phase 5 consent import (application/contact_permissions.py). source_record_id is
    # the id the source system used for this record (nullable: the original single-record
    # POST never had one). ingestion_run_id links it to the import run that wrote it.
    # content_hash lets an import be re-run safely: a byte-identical record (same
    # customer, channel, status, source, reason, effective_at) is a no-op, not a new row.
    source_record_id = Column(String, nullable=True)
    ingestion_run_id = Column(String, index=True, nullable=True)
    content_hash = Column(String, index=True, nullable=True)

    __table_args__ = (
        UniqueConstraint("customer_id", "content_hash", name="uq_contact_permission_content"),
    )


class EventIdentityAliasModel(Base):
    """An additional identity under which an EXISTING event is recognised.

    Written only by identity reconciliation (application/identity_reconciliation.py) for
    events stored under an older identity scheme. The event row itself is never
    rewritten: its original external_system / external_event_id are kept, and the alias
    lets deduplication recognise the same row when it arrives under the current scheme.
    """

    __tablename__ = "event_identity_aliases"

    id = Column(String, primary_key=True)
    event_id = Column(String, ForeignKey("revenue_events.event_id"), index=True, nullable=False)
    alias_system = Column(String, nullable=False)
    alias_event_id = Column(String, nullable=False)
    method = Column(String, nullable=False)  # which legacy scheme it was derived from
    details = Column(JSON, nullable=True)
    reconciliation_run_id = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint("alias_system", "alias_event_id", name="uq_event_identity_alias"),
    )


class OutcomeEventModel(Base):
    """An authoritative outcome event: money received, an obligation cancelled, or a
    promise-to-pay created/broken. See application/outcome_events.py.

    Append-only and the sole source of truth for recovered/outstanding/cancelled money;
    CaseModel.recovered_amount etc. are a materialised cache written only from here. A row
    is written exactly once, at ingestion, with its match and reconciliation result
    already decided - there is no separate "apply later" step, so nothing here is ever
    retroactively recomputed by a later event (see the module docstring for why: a
    back-dated event must not rewrite an amount already reported as recovered)."""

    __tablename__ = "outcome_events"

    outcome_event_id = Column(String, primary_key=True)
    external_system = Column(String, nullable=False)
    external_event_id = Column(String, nullable=False)  # dedup key with external_system

    domain = Column(String, nullable=False)  # RiskCategory value
    kind = Column(String, nullable=False)  # OutcomeKind value
    outcome_type = Column(String, nullable=False)  # the source vocabulary value, normalised

    customer_id = Column(String, nullable=False, index=True)
    customer_system = Column(String, nullable=True)
    reference_id = Column(String, nullable=True, index=True)

    amount = Column(Numeric(precision=18, scale=4), nullable=True)
    currency = Column(String, nullable=True)
    occurred_at = Column(DateTime, nullable=False)  # the business/settlement timestamp
    promised_date = Column(DateTime, nullable=True)  # PROMISE_CREATED only

    raw_payload = Column(JSON, nullable=True)

    source_type = Column(String, nullable=True)  # application.ingestion.SourceType
    source_ref = Column(String, index=True, nullable=True)
    ingestion_run_id = Column(String, index=True, nullable=True)
    ingested_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    # Matching and reconciliation result, decided once at ingestion.
    match_status = Column(String, nullable=False)  # MATCHED | UNMATCHED
    match_reason = Column(String, nullable=True)  # set when UNMATCHED
    case_id = Column(String, ForeignKey("recovery_cases.case_id"), nullable=True, index=True)
    applied_amount = Column(Numeric(precision=18, scale=4), nullable=True)
    excess_amount = Column(Numeric(precision=18, scale=4), nullable=True)
    effect = Column(String, nullable=True)  # APPLIED | EXCESS | NO_EFFECT | AFTER_CLOSURE

    # Phase 8A: where this outcome event actually came from, so a synthetic/test event
    # can never silently be counted as authoritative production evidence (application
    # .outcome_provenance.OutcomeProvenance). Not nullable, no implicit default at this
    # layer - every caller must say what it is; application.outcome_events defaults an
    # omitted value to TEST (never PRODUCTION) before it reaches this column.
    provenance = Column(String, nullable=False, index=True)
    # Phase 9: required for provenance=human_confirmed (a pointer to something OUTSIDE
    # RecoverChain's own generated ids - a ticket, a call log, an email thread - never
    # one of RecoverChain's own oc_/case_/cobs_/evt_ ids). Required, and validated
    # against a registered source, for the per-integration provenances. See
    # application/evidence_sources.py.
    evidence_reference = Column(String, nullable=True)
    source_id = Column(String, ForeignKey("evidence_sources.source_id"), nullable=True, index=True)

    __table_args__ = (
        UniqueConstraint(
            "external_system", "external_event_id", name="uq_outcome_event_external_id"
        ),
    )


class PromiseModel(Base):
    """A promise-to-pay, created by a PROMISE_CREATED outcome event. Never itself money:
    the financial outcome is the matching settlement event. Status is derived on read
    from the case's applied settlements and occurred_at (application/outcome_events.py);
    only PROMISE_BROKEN, when the source reports one explicitly, is stored."""

    __tablename__ = "promises"

    promise_id = Column(String, primary_key=True)
    outcome_event_id = Column(
        String, ForeignKey("outcome_events.outcome_event_id"), nullable=False, unique=True
    )
    case_id = Column(String, ForeignKey("recovery_cases.case_id"), nullable=False, index=True)
    domain = Column(String, nullable=False)
    customer_id = Column(String, nullable=False, index=True)
    reference_id = Column(String, nullable=True)

    promised_amount = Column(Numeric(precision=18, scale=4), nullable=False)
    currency = Column(String, nullable=False)
    promised_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, nullable=False)  # occurred_at of PROMISE_CREATED

    broken_reported_at = Column(DateTime, nullable=True)  # set by an explicit PROMISE_BROKEN


class CauseObservationModel(Base):
    """Phase 8A: an independently observed root-cause fact, distinct from the
    deterministic diagnosis rule's OUTPUT (RootCauseDiagnosis, application
    .diagnosis_engine.py). The diagnosis engine's cause_category is *derived* from the
    same failure_code/reason text a future classifier would use as input - training on
    it would just re-learn the existing lookup table (see docs/history/
    phase7_ai_ml_audit.md Step 1). A row here is instead an INDEPENDENT fact from
    outside that rule: a human operator confirming what actually happened, or (in a
    future integration) a payment-gateway/invoicing/subscription/checkout system's own
    report of the reason. Only a row here can ever become a legitimate label for a
    future root-cause classifier; RootCauseDiagnosis.cause_category never can."""

    __tablename__ = "cause_observations"

    observation_id = Column(String, primary_key=True)
    case_id = Column(String, ForeignKey("recovery_cases.case_id"), nullable=False, index=True)
    observed_cause_category = Column(String, nullable=False)  # a RootCauseCategory value
    # Where the observation came from - never "rule" or "diagnosis_engine": those
    # aren't observations, they're what this table exists to be independent OF.
    source = Column(String, nullable=False)
    note = Column(String, nullable=True)
    # Who/what recorded it - an operator's identity or an integration's own id.
    recorded_by = Column(String, nullable=True)
    # Phase 8B: a pointer into the evidence itself (a support-ticket id, a call log
    # reference, an export file's row id) - so an observation can be traced back to
    # what it was actually based on, not just trusted on its own say-so.
    evidence_reference = Column(String, nullable=True)
    # Phase 8B: optional observer-stated confidence, 0.0-1.0. Never required - a human
    # confirming what happened is not obligated to quantify certainty.
    confidence = Column(Numeric(precision=4, scale=3), nullable=True)
    recorded_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))


class MLDatasetSnapshotModel(Base):
    """Phase 8B: an immutable, checksummed record of exactly which ELIGIBLE cases a
    future ML experiment would train/evaluate on, and why everything else was
    excluded. Rows here are never updated - a re-run that would change the answer
    creates a new (dataset_name, version) instead, so a past experiment's dataset_id
    + version always resolves to the exact same case list forever. See
    application/dataset_snapshot.py and application/eligibility.py."""

    __tablename__ = "ml_dataset_snapshots"

    snapshot_id = Column(String, primary_key=True)
    dataset_name = Column(String, nullable=False, index=True)
    version = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    created_by = Column(String, nullable=True)

    eligible_case_count = Column(Integer, nullable=False)
    provenance_distribution = Column(JSON, nullable=True)
    time_range_start = Column(DateTime, nullable=True)
    time_range_end = Column(DateTime, nullable=True)
    feature_schema_version = Column(String, nullable=False)
    label_schema_version = Column(String, nullable=False)
    exclusion_counts = Column(JSON, nullable=True)
    eligible_case_ids = Column(JSON, nullable=False)
    checksum = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint("dataset_name", "version", name="uq_ml_dataset_snapshot_name_version"),
    )


class EvidenceSourceModel(Base):
    """Phase 9: a registered external evidence source. Registration alone never grants
    trust - a source only becomes usable for its registered provenance once
    `integration_status` reaches VERIFIED, which in turn requires real credential
    material this environment does not have (application/evidence_sources.py). A
    client cannot make itself a trusted production source merely by sending a
    provenance value on an outcome event; it must be registered here first, and the
    registration itself must pass verification, which is a real (not decorative)
    check."""

    __tablename__ = "evidence_sources"

    source_id = Column(String, primary_key=True)
    source_type = Column(
        String, nullable=False, index=True
    )  # e.g. payment_provider, invoice_system
    provider = Column(String, nullable=False)  # e.g. "stripe", "netsuite" - free text label
    domain = Column(String, nullable=False)  # RiskCategory this source can report outcomes for
    # NOT_CONFIGURED (no credential material present) | VERIFIED (credential material
    # present and a real check passed) - never any other value; see
    # application/evidence_sources.py::check_integration_status.
    integration_status = Column(String, nullable=False)
    schema_version = Column(String, nullable=False)
    enabled = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    last_verified_at = Column(DateTime, nullable=True)
    registered_by = Column(String, nullable=True)


class CustomerIdentityLinkModel(Base):
    """A deterministic mapping from a customer id in an external system to RecoverChain's
    customer_id. Written only by an explicit POST; never inferred (no name/email
    matching). Used only to resolve `customer_system` on an outcome or consent-import
    record - an unlinked (system, external id) pair is UNMATCHED / UNKNOWN, never
    guessed."""

    __tablename__ = "customer_identity_links"

    id = Column(String, primary_key=True)
    system = Column(String, nullable=False)
    external_customer_id = Column(String, nullable=False)
    customer_id = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint("system", "external_customer_id", name="uq_customer_identity_link"),
    )


class EvaluationRunModel(Base):
    """Result of one evaluation-harness run.

    Evaluation scenarios never touch recovery_cases / revenue_events / audit_records /
    execution_attempts: they run in an isolated ephemeral store (see
    evaluation.runner.EvaluationRunner.isolated) and only this summary row is kept.
    No operational aggregate reads this table, so evaluation data cannot reach a
    dashboard figure by construction rather than by filtering."""

    __tablename__ = "evaluation_runs"

    run_id = Column(String, primary_key=True)
    status = Column(String, nullable=False)  # COMPLETED | FAILED
    experiment_mode = Column(String, nullable=False)
    isolation = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=False, index=True)
    completed_at = Column(DateTime, nullable=True)
    total_scenarios = Column(Integer, nullable=True)
    passed_scenarios = Column(Integer, nullable=True)
    metrics = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)


# Phase 17: register the append-only guards for evidence tables (see infrastructure/immutability.py).
from infrastructure import immutability  # noqa: E402,F401
