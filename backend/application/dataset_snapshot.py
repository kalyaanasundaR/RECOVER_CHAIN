"""Phase 8B — immutable, checksummed dataset snapshots (Step 9).

A snapshot freezes exactly which case_ids are ELIGIBLE (application/eligibility.py)
at one moment, plus enough metadata for a future ML experiment to know precisely what
it trained/evaluated on. Rows are NEVER updated: creating a snapshot with the same
`dataset_name` again always mints the next `version`, never overwrites a prior one, so
a past experiment's (dataset_name, version) always resolves to the same case list.

This module only reads the operational store (cases, outcome_events, quality checks)
and writes to its own, separate ml_dataset_snapshots table - it never mutates a
RecoveryCase, an OutcomeEventModel row, or anything else operational (Step 10).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from application.eligibility import (
    FEATURE_SCHEMA_VERSION,
    LABEL_SCHEMA_VERSION,
    EligibilityStatus,
    evaluate_all,
)
from infrastructure.orm import MLDatasetSnapshotModel, OutcomeEventModel


def _compute_checksum(eligible_case_ids: list[str], feature_schema: str, label_schema: str) -> str:
    """Deterministic over the frozen inputs only - re-running this function on the
    exact same case list and schema versions always produces the exact same hash,
    which is the actual reproducibility guarantee (not just "a hash exists")."""
    canonical = json.dumps(
        {
            "eligible_case_ids": sorted(eligible_case_ids),
            "feature_schema_version": feature_schema,
            "label_schema_version": label_schema,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_snapshot(db: Session, *, dataset_name: str, created_by: str | None = None) -> dict:
    results = evaluate_all(db)
    eligible = [r for r in results if r.status == EligibilityStatus.ELIGIBLE]
    exclusion_counts: dict[str, int] = {}
    for r in results:
        if r.status != EligibilityStatus.ELIGIBLE:
            for reason in r.reasons:
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    eligible_case_ids = sorted(r.case_id for r in eligible)

    provenance_distribution: dict[str, int] = {}
    time_range_start = time_range_end = None
    if eligible_case_ids:
        outcomes = (
            db.query(OutcomeEventModel)
            .filter(
                OutcomeEventModel.case_id.in_(eligible_case_ids),
                OutcomeEventModel.match_status == "MATCHED",
            )
            .all()
        )
        for o in outcomes:
            provenance_distribution[o.provenance] = provenance_distribution.get(o.provenance, 0) + 1
            if o.occurred_at is not None:
                if time_range_start is None or o.occurred_at < time_range_start:
                    time_range_start = o.occurred_at
                if time_range_end is None or o.occurred_at > time_range_end:
                    time_range_end = o.occurred_at

    checksum = _compute_checksum(eligible_case_ids, FEATURE_SCHEMA_VERSION, LABEL_SCHEMA_VERSION)

    # Next version for this dataset_name - never reused, never overwritten.
    existing_max = (
        db.query(MLDatasetSnapshotModel)
        .filter(MLDatasetSnapshotModel.dataset_name == dataset_name)
        .order_by(MLDatasetSnapshotModel.version.desc())
        .first()
    )
    next_version = (existing_max.version + 1) if existing_max else 1

    row = MLDatasetSnapshotModel(
        snapshot_id=f"snap_{uuid.uuid4().hex[:12]}",
        dataset_name=dataset_name,
        version=next_version,
        created_at=datetime.now(UTC).replace(tzinfo=None),
        created_by=created_by,
        eligible_case_count=len(eligible_case_ids),
        provenance_distribution=provenance_distribution,
        time_range_start=time_range_start,
        time_range_end=time_range_end,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        label_schema_version=LABEL_SCHEMA_VERSION,
        exclusion_counts=exclusion_counts,
        eligible_case_ids=eligible_case_ids,
        checksum=checksum,
    )
    db.add(row)
    db.flush()
    return snapshot_dict(row)


def snapshot_dict(row: MLDatasetSnapshotModel) -> dict:
    return {
        "snapshot_id": row.snapshot_id,
        "dataset_name": row.dataset_name,
        "version": row.version,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "created_by": row.created_by,
        "eligible_case_count": row.eligible_case_count,
        "provenance_distribution": row.provenance_distribution,
        "time_range_start": row.time_range_start.isoformat() if row.time_range_start else None,
        "time_range_end": row.time_range_end.isoformat() if row.time_range_end else None,
        "feature_schema_version": row.feature_schema_version,
        "label_schema_version": row.label_schema_version,
        "exclusion_counts": row.exclusion_counts,
        "eligible_case_ids": row.eligible_case_ids,
        "checksum": row.checksum,
    }
