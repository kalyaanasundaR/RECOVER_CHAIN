"""cause-observation evidence fields and immutable dataset snapshots

Phase 8B.

- cause_observations gains `evidence_reference` (a pointer into what the observation
  was actually based on) and `confidence` (an optional observer-stated 0.0-1.0 value).
- new table `ml_dataset_snapshots`: an immutable, checksummed record of exactly which
  ELIGIBLE cases a future ML experiment would train on, and why the rest were
  excluded. See application/dataset_snapshot.py and application/eligibility.py, and
  docs/history/PHASE_8B_INDEPENDENT_EVIDENCE_REPORT.md Step 8/9.

All additive; existing rows upgrade in place.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-09-19

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("cause_observations") as batch_op:
        batch_op.add_column(sa.Column("evidence_reference", sa.String(), nullable=True))
        batch_op.add_column(
            sa.Column("confidence", sa.Numeric(precision=4, scale=3), nullable=True)
        )

    op.create_table(
        "ml_dataset_snapshots",
        sa.Column("snapshot_id", sa.String(), primary_key=True),
        sa.Column("dataset_name", sa.String(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("eligible_case_count", sa.Integer(), nullable=False),
        sa.Column("provenance_distribution", sa.JSON(), nullable=True),
        sa.Column("time_range_start", sa.DateTime(), nullable=True),
        sa.Column("time_range_end", sa.DateTime(), nullable=True),
        sa.Column("feature_schema_version", sa.String(), nullable=False),
        sa.Column("label_schema_version", sa.String(), nullable=False),
        sa.Column("exclusion_counts", sa.JSON(), nullable=True),
        sa.Column("eligible_case_ids", sa.JSON(), nullable=False),
        sa.Column("checksum", sa.String(), nullable=False),
        sa.UniqueConstraint("dataset_name", "version", name="uq_ml_dataset_snapshot_name_version"),
    )
    op.create_index(
        "ix_ml_dataset_snapshots_dataset_name",
        "ml_dataset_snapshots",
        ["dataset_name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_ml_dataset_snapshots_dataset_name", table_name="ml_dataset_snapshots")
    op.drop_table("ml_dataset_snapshots")

    with op.batch_alter_table("cause_observations") as batch_op:
        batch_op.drop_column("confidence")
        batch_op.drop_column("evidence_reference")
