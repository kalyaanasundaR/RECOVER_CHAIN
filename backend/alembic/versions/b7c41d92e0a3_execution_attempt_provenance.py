"""execution attempt provenance (execution_id, agent_type)

Phase 2: CaseModel.execution_record holds only the LAST execution, so a case with
several attempts reported just the most recent one and the earlier real executions
were unrecoverable from the case row. These two additive, nullable columns let the
append-only execution_attempts ledger answer "which agent did what, when, and under
whose authorization" on its own.

Revision ID: b7c41d92e0a3
Revises: 6aaf57d8d4b5
Create Date: 2026-09-13

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b7c41d92e0a3'
down_revision: Union[str, Sequence[str], None] = '6aaf57d8d4b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('execution_attempts', sa.Column('execution_id', sa.String(), nullable=True))
    op.add_column('execution_attempts', sa.Column('agent_type', sa.String(), nullable=True))
    op.create_index(
        op.f('ix_execution_attempts_execution_id'),
        'execution_attempts',
        ['execution_id'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_execution_attempts_execution_id'), table_name='execution_attempts')
    op.drop_column('execution_attempts', 'agent_type')
    op.drop_column('execution_attempts', 'execution_id')
