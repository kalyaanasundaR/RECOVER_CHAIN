"""optimistic concurrency version on recovery_cases

Phase 2: the repository does a read-modify-write with no version check, and
with_for_update() is a no-op on SQLite (the dialect drops FOR UPDATE), so the last
committer won. A stage request that read a case just before verification committed
could write POLICY_EVALUATED back over FULLY_RECOVERED. With version_id_col set,
SQLAlchemy adds "WHERE version_id = :read_version" to every UPDATE and raises
StaleDataError instead, which the API surfaces as HTTP 409.

Additive and backfilled via server_default, so existing rows upgrade in place.

Revision ID: c8d5e0a41f27
Revises: b7c41d92e0a3
Create Date: 2026-09-13

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c8d5e0a41f27'
down_revision: Union[str, Sequence[str], None] = 'b7c41d92e0a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'recovery_cases',
        sa.Column('version_id', sa.Integer(), nullable=False, server_default='1'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('recovery_cases', 'version_id')
