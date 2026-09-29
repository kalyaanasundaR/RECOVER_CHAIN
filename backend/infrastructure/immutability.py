"""Phase 17 - ORM-level append-only guards for evidence tables.

Approval evidence, policy-revalidation evidence and every other audit record live in
`audit_records`; executions live in `execution_attempts`; authoritative outcomes in
`outcome_events`. None of them may be rewritten. Corrections are made by appending a new
record, never by editing history.

  audit_records       no UPDATE, no DELETE
  execution_attempts  no UPDATE, no DELETE
  ml_dataset_snapshots no UPDATE, no DELETE (already documented as immutable)
  outcome_events      no DELETE; identity/evidence columns (provenance, evidence_reference,
                      source_id, external ids, kind, type, amount, timestamps, customer)
                      never change; the match/reconciliation columns may change ONLY when
                      the row was UNMATCHED (the existing rematch flow) - an already MATCHED
                      row is frozen.

Scope, stated plainly: these are SQLAlchemy mapper events. They stop every code path in this
application (and any code using the ORM) from mutating history, and tests prove it. They do
NOT stop a person with direct database access, a raw SQL statement, or a bulk
`query.update()/delete()`; that would need database-level triggers or permissions.
"""

from __future__ import annotations

from sqlalchemy import event, inspect

from infrastructure.orm import (
    AuditModel,
    ExecutionAttemptModel,
    MLDatasetSnapshotModel,
    OutcomeEventModel,
)


class ImmutableRecordError(RuntimeError):
    """An attempt to rewrite append-only evidence."""


_OUTCOME_FROZEN = (
    "provenance",
    "evidence_reference",
    "source_id",
    "external_system",
    "external_event_id",
    "kind",
    "outcome_type",
    "domain",
    "customer_id",
    "amount",
    "currency",
    "occurred_at",
    "raw_payload",
    "ingested_at",
    "outcome_event_id",
)
_OUTCOME_MATCH = (
    "match_status",
    "match_reason",
    "case_id",
    "applied_amount",
    "excess_amount",
    "effect",
)


def _forbid(kind: str):
    def _guard(mapper, connection, target):
        raise ImmutableRecordError(f"{kind} is append-only: it cannot be modified or deleted")

    return _guard


for _model, _name in (
    (AuditModel, "audit record"),
    (ExecutionAttemptModel, "execution ledger row"),
    (MLDatasetSnapshotModel, "dataset snapshot"),
):
    event.listen(_model, "before_update", _forbid(_name))
    event.listen(_model, "before_delete", _forbid(_name))


@event.listens_for(OutcomeEventModel, "before_delete")
def _outcome_no_delete(mapper, connection, target):
    raise ImmutableRecordError("outcome event is append-only: it cannot be deleted")


@event.listens_for(OutcomeEventModel, "before_update")
def _outcome_frozen(mapper, connection, target):
    state = inspect(target)
    for col in _OUTCOME_FROZEN:
        if state.attrs[col].history.has_changes():
            raise ImmutableRecordError(f"outcome event column {col!r} is immutable evidence")
    match_hist = state.attrs["match_status"].history
    was_matched = (
        match_hist.deleted[0] if match_hist.deleted else target.match_status
    ) == "MATCHED"
    if was_matched and any(state.attrs[c].history.has_changes() for c in _OUTCOME_MATCH):
        raise ImmutableRecordError("a MATCHED outcome event cannot be re-matched or re-applied")
