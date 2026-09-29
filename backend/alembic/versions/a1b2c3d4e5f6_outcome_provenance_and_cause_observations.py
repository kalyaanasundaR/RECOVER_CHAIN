"""outcome provenance and cause observations

Phase 8A.

- outcome_events gains `provenance` (String, NOT NULL). Every existing row in this
  dev DB is self-generated test/verification data (Phase 5/6/7), so the backfill
  default for pre-existing rows is 'test' - never 'production', because none of them
  are. Every new row must state its provenance explicitly (application.outcome_events
  defaults an omitted value to 'test' at the application layer too - this column has
  no implicit "assume real" path anywhere).
- new table `cause_observations`: an independently-observed root-cause fact, distinct
  from the deterministic diagnosis engine's rule-derived cause_category. See
  infrastructure/orm.py::CauseObservationModel and docs/history/
  PHASE_8A_DATA_READINESS_REPORT.md Step 8.

All additive; existing rows upgrade in place.

Revision ID: a1b2c3d4e5f6
Revises: f3a9c7d1b284
Create Date: 2026-09-17

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f3a9c7d1b284"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("outcome_events") as batch_op:
        batch_op.add_column(
            sa.Column("provenance", sa.String(), nullable=False, server_default="test")
        )
    op.create_index(
        "ix_outcome_events_provenance", "outcome_events", ["provenance"], unique=False
    )

    op.create_table(
        "cause_observations",
        sa.Column("observation_id", sa.String(), primary_key=True),
        sa.Column(
            "case_id", sa.String(), sa.ForeignKey("recovery_cases.case_id"), nullable=False
        ),
        sa.Column("observed_cause_category", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("note", sa.String(), nullable=True),
        sa.Column("recorded_by", sa.String(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_cause_observations_case_id", "cause_observations", ["case_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_cause_observations_case_id", table_name="cause_observations")
    op.drop_table("cause_observations")

    op.drop_index("ix_outcome_events_provenance", table_name="outcome_events")
    with op.batch_alter_table("outcome_events") as batch_op:
        batch_op.drop_column("provenance")
