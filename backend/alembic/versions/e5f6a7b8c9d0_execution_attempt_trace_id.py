"""execution attempt trace_id

Phase 16: the append-only execution ledger row now carries the journey `trace_id`
(application/trace_ids.py) so an execution can be tied to its agent/policy/approval/
outcome from the ledger alone. Additive and nullable; pre-Phase-16 rows stay NULL and
are reported as missing-trace by the validator, never back-filled.

Revision ID: e5f6a7b8c9d0
Revises: c3d4e5f6a7b8
Create Date: 2026-09-19

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, Sequence[str], None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('execution_attempts', sa.Column('trace_id', sa.String(), nullable=True))
    op.create_index(
        op.f('ix_execution_attempts_trace_id'), 'execution_attempts', ['trace_id'], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_execution_attempts_trace_id'), table_name='execution_attempts')
    op.drop_column('execution_attempts', 'trace_id')
