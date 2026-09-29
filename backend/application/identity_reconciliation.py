"""Re-identification of events stored under an older identity scheme.

The problem. File rows have had three identities:

- pre-Phase 3: ``external_system="Dataset-<dataset_id>"``, ``external_event_id="row_<n>"``
  - identity of the UPLOAD, not of the row. The same row uploaded again was a new event.
- Phase 3 (v1): ``file_upload`` / ``fp:<hash>:<n>`` - content identity, but the hash
  included the risk category the retired keyword heuristic assigned.
- Phase 4 (v2): ``file_upload`` / ``fp2:<hash>:<n>`` - content identity without the
  category (application/ingestion.py).

v1 ids are recognised at ingestion time directly. Pre-Phase-3 ids cannot be: nothing in
"row_5" says what the row contained. So the first upload of an already-ingested file
under a newer scheme created duplicate cases.

The solution. A reconciliation run computes, for every older event, the v2 identity its
row has today, from the fields stored on the event, and records it as an ALIAS
(event_identity_aliases). Deduplication honours aliases, so the next upload of the same
row is recognised. The run:

- never rewrites or deletes an event, case or audit record (the original external ids
  stay on the event);
- never merges anything: when two events claim the same identity, or the identity is
  already used by another event, it is reported as a conflict for a person to review;
- is deterministic (events are processed in a fixed order) and idempotent (an alias
  that already exists is counted, not written again);
- is a dry run unless ``apply=True``; an applied run that adds aliases writes one audit
  record, and every alias carries the run id.

The limitations are listed in LIMITATIONS and returned with every report.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from application.ingestion import (
    FILE_EXTERNAL_SYSTEM,
    ISO_4217,
    derive_failure_code,
    event_fingerprint,
    event_fingerprint_v2,
    fingerprint_event_id_v2,
)
from infrastructure.orm import AuditModel, EventIdentityAliasModel, EventModel

METHOD_LEGACY_DATASET_ROW = "legacy_dataset_row"
METHOD_CONTENT_V1 = "content_identity_v1"
ALIAS_SYSTEM = FILE_EXTERNAL_SYSTEM
MAX_LISTED = 100
_IN_CHUNK = 500

LIMITATIONS = [
    "Pre-Phase-3 events stored no identity of their row. Their identity is recomputed from "
    "the stored customer, amount, currency, timestamp, reference and payload. A stored "
    "timestamp carries no offset and is taken as UTC, so a row whose source used another "
    "offset can fail to match a re-upload; it is then simply not recognised (it is never "
    "matched by guesswork).",
    "A pre-Phase-3 reference of the form tx_<row number> was fabricated by the old "
    "ingestion and is ignored; a real reference from the file is kept.",
    "Occurrence numbers for pre-Phase-3 rows are assigned in row order within each original "
    "upload, which is what a re-upload of the same file computes. A file edited between "
    "uploads (an identical row inserted earlier) can shift them.",
    "A v1 event is re-identified only when its stored fields reproduce its original v1 "
    "fingerprint exactly; otherwise it is reported as not re-identifiable.",
    "Two older events with the same identity (typically the same file uploaded twice under "
    "the pre-Phase-3 scheme) are a conflict. They are never merged: the duplicate cases "
    "they created remain for a person to review.",
]

_ROW_ID = re.compile(r"^row_(\d+)$")
_FABRICATED_REF = re.compile(r"^tx_(\d+)$")
_V1_ID = re.compile(r"^fp:([0-9a-f]{32}):(\d+)$")


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _inputs(e: EventModel) -> tuple[dict | None, str | None]:
    currency = (e.currency or "").strip().upper()
    if currency not in ISO_4217:
        return None, "stored currency is not an ISO 4217 code"
    if not e.customer_id or e.amount is None or e.timestamp is None:
        return None, "customer, amount or timestamp is missing"
    return {
        "customer_id": e.customer_id,
        "amount": Decimal(str(e.amount)),
        "currency": currency,
        "timestamp": _aware(e.timestamp),
        "failure_code": derive_failure_code(e.raw_payload or {}),
    }, None


def _chunks(items: list, size: int = _IN_CHUNK):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def reconcile_identities(db, *, apply: bool = False, dataset_ids: list[str] | None = None) -> dict:
    """``dataset_ids`` limits the run to the events of those datasets; conflicts are still
    checked against every stored event and alias."""
    run_id = f"idr_{uuid.uuid4().hex[:16]}"
    candidates: list[tuple[EventModel, str, str, dict]] = []
    not_reidentifiable: list[dict] = []

    legacy_q = db.query(EventModel).filter(EventModel.external_system.like("Dataset-%"))
    v1_q = db.query(EventModel).filter(
        EventModel.external_system == FILE_EXTERNAL_SYSTEM,
        EventModel.external_event_id.like("fp:%"),
    )
    if dataset_ids:
        legacy_q = legacy_q.filter(
            EventModel.external_system.in_([f"Dataset-{d}" for d in dataset_ids])
        )
        v1_q = v1_q.filter(EventModel.source_ref.in_(dataset_ids))
    legacy = legacy_q.all()
    v1 = v1_q.all()

    # -- pre-Phase-3 dataset rows, per original upload, in row order -------------------
    uploads: dict[str, list[tuple[int, EventModel]]] = {}
    for e in legacy:
        m = _ROW_ID.match(e.external_event_id or "")
        if not m:
            not_reidentifiable.append(
                {"event_id": e.event_id, "reason": "legacy id is not of the form row_<n>"}
            )
            continue
        uploads.setdefault(e.external_system, []).append((int(m.group(1)), e))

    for upload in sorted(uploads):
        occurrences: dict[str, int] = {}
        for row_index, e in sorted(uploads[upload], key=lambda t: (t[0], t[1].event_id)):
            inputs, problem = _inputs(e)
            if problem:
                not_reidentifiable.append({"event_id": e.event_id, "reason": problem})
                continue
            ref = e.reference_id
            fabricated = bool(
                ref and (fm := _FABRICATED_REF.match(ref)) and int(fm.group(1)) == row_index
            )
            fp = event_fingerprint_v2(**inputs, reference_id=None if fabricated else ref)
            occurrence = occurrences.get(fp, 0)
            occurrences[fp] = occurrence + 1
            candidates.append(
                (
                    e,
                    fingerprint_event_id_v2(fp, occurrence),
                    METHOD_LEGACY_DATASET_ROW,
                    {
                        "original_external_system": e.external_system,
                        "original_external_event_id": e.external_event_id,
                        "fabricated_reference_ignored": ref if fabricated else None,
                    },
                )
            )

    # -- Phase 3 v1 content identity, verified against the stored fingerprint ------------
    for e in sorted(v1, key=lambda ev: ev.external_event_id):
        m = _V1_ID.match(e.external_event_id or "")
        if not m:
            not_reidentifiable.append(
                {"event_id": e.event_id, "reason": "v1 id is not of the form fp:<hash>:<n>"}
            )
            continue
        inputs, problem = _inputs(e)
        if problem:
            not_reidentifiable.append({"event_id": e.event_id, "reason": problem})
            continue
        reproduced = event_fingerprint(
            **inputs, risk_category=e.risk_category, reference_id=e.reference_id
        )
        if reproduced != m.group(1):
            not_reidentifiable.append(
                {
                    "event_id": e.event_id,
                    "reason": "stored fields do not reproduce the original v1 fingerprint",
                }
            )
            continue
        code = inputs["failure_code"]
        v2_inputs = {**inputs, "failure_code": None if code == "nan" else code}
        fp = event_fingerprint_v2(**v2_inputs, reference_id=e.reference_id)
        candidates.append(
            (
                e,
                fingerprint_event_id_v2(fp, int(m.group(2))),
                METHOD_CONTENT_V1,
                {"original_external_event_id": e.external_event_id},
            )
        )

    # -- classify against what already exists -------------------------------------------
    alias_ids = [c[1] for c in candidates]
    alias_owner: dict[str, str] = {}
    identity_owner: dict[str, tuple[str, str]] = {}
    for chunk in _chunks(alias_ids):
        for a in (
            db.query(EventIdentityAliasModel)
            .filter(
                EventIdentityAliasModel.alias_system == ALIAS_SYSTEM,
                EventIdentityAliasModel.alias_event_id.in_(chunk),
            )
            .all()
        ):
            alias_owner[a.alias_event_id] = a.event_id
        for ext_id, event_id, case_id in (
            db.query(EventModel.external_event_id, EventModel.event_id, EventModel.case_id)
            .filter(
                EventModel.external_system == ALIAS_SYSTEM,
                EventModel.external_event_id.in_(chunk),
            )
            .all()
        ):
            identity_owner[ext_id] = (event_id, case_id)

    claimed: dict[str, EventModel] = {}
    eligible: list[tuple[EventModel, str, str, dict]] = []
    already_aliased = 0
    conflicts: list[dict] = []
    for e, alias, method, details in candidates:
        base = {"event_id": e.event_id, "case_id": e.case_id, "identity": alias, "method": method}
        if alias_owner.get(alias) == e.event_id:
            already_aliased += 1
        elif alias in alias_owner:
            conflicts.append(
                {
                    **base,
                    "kind": "identity_aliased_to_another_event",
                    "other_event_id": alias_owner[alias],
                }
            )
        elif alias in identity_owner:
            other_event, other_case = identity_owner[alias]
            conflicts.append(
                {
                    **base,
                    "kind": "identity_used_by_a_newer_event",
                    "other_event_id": other_event,
                    "other_case_id": other_case,
                }
            )
        elif alias in claimed:
            conflicts.append(
                {
                    **base,
                    "kind": "same_identity_as_another_older_event",
                    "other_event_id": claimed[alias].event_id,
                    "other_case_id": claimed[alias].case_id,
                }
            )
        else:
            claimed[alias] = e
            eligible.append((e, alias, method, details))

    added = 0
    if apply and eligible:
        now = datetime.now(UTC).replace(tzinfo=None)
        for e, alias, method, details in eligible:
            db.add(
                EventIdentityAliasModel(
                    id=uuid.uuid4().hex,
                    event_id=e.event_id,
                    alias_system=ALIAS_SYSTEM,
                    alias_event_id=alias,
                    method=method,
                    details=details,
                    reconciliation_run_id=run_id,
                    created_at=now,
                )
            )
            added += 1
        db.add(
            AuditModel(
                id=uuid.uuid4().hex,
                case_id=None,
                from_state="N/A",
                to_state="N/A",
                evidence={
                    "action": "identity_reconciliation",
                    "run_id": run_id,
                    "aliases_added": added,
                    "already_aliased": already_aliased,
                    "conflicts": len(conflicts),
                    "not_reidentifiable": len(not_reidentifiable),
                },
            )
        )
        db.commit()

    return {
        "run_id": run_id,
        "mode": "APPLIED" if apply else "DRY_RUN",
        "scope": sorted(dataset_ids) if dataset_ids else "ALL",
        "scanned": {METHOD_LEGACY_DATASET_ROW: len(legacy), METHOD_CONTENT_V1: len(v1)},
        "eligible": len(eligible),
        "aliases_added": added,
        "already_aliased": already_aliased,
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:MAX_LISTED],
        "not_reidentifiable_count": len(not_reidentifiable),
        "not_reidentifiable": not_reidentifiable[:MAX_LISTED],
        "eligible_sample": [
            {"event_id": e.event_id, "case_id": e.case_id, "identity": alias, "method": method}
            for e, alias, method, _ in eligible[:MAX_LISTED]
        ],
        "limitations": LIMITATIONS,
    }
