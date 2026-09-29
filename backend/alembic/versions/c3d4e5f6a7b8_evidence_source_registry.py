"""evidence source registry and outcome evidence_reference/source_id

Phase 9.

- new table `evidence_sources`: a registered external evidence source. Registration
  alone never grants trust - see application/evidence_sources.py and infrastructure/
  orm.py::EvidenceSourceModel.
- outcome_events gains `evidence_reference` (required for provenance=human_confirmed,
  a pointer to something outside RecoverChain's own generated ids) and `source_id`
  (FK to evidence_sources, required for the per-integration provenances).

All additive; existing rows upgrade in place (both new outcome_events columns are
nullable - historical rows predate this requirement and are left as they are).

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-09-18

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "evidence_sources",
        sa.Column("source_id", sa.String(), primary_key=True),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("domain", sa.String(), nullable=False),
        sa.Column("integration_status", sa.String(), nullable=False),
        sa.Column("schema_version", sa.String(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_verified_at", sa.DateTime(), nullable=True),
        sa.Column("registered_by", sa.String(), nullable=True),
    )
    op.create_index(
        "ix_evidence_sources_source_type", "evidence_sources", ["source_type"], unique=False
    )

    with op.batch_alter_table("outcome_events") as batch_op:
        batch_op.add_column(sa.Column("evidence_reference", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("source_id", sa.String(), nullable=True))
        batch_op.create_foreign_key(
            "fk_outcome_events_source_id", "evidence_sources", ["source_id"], ["source_id"]
        )
    op.create_index(
        "ix_outcome_events_source_id", "outcome_events", ["source_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_outcome_events_source_id", table_name="outcome_events")
    with op.batch_alter_table("outcome_events") as batch_op:
        batch_op.drop_constraint("fk_outcome_events_source_id", type_="foreignkey")
        batch_op.drop_column("source_id")
        batch_op.drop_column("evidence_reference")

    op.drop_index("ix_evidence_sources_source_type", table_name="evidence_sources")
    op.drop_table("evidence_sources")
