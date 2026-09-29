"""domain semantics: dataset category contract, contact permissions, identity aliases

Phase 4.

- datasets gain the category contract: declared_domain, source_system and the last
  domain_classification (with its evidence). Schema, version and runs already exist
  (recoverchain_signals, content_sha256, ingestion_runs).
- revenue_events.domain_basis records why each event carries its risk_category.
- ingestion_runs.domain_classification records how a run's rows got their domain.
- execution_attempts gain policy_decision_id and contact_permission, so contact
  history answers "under which decision?" and "what was permitted then?" from the
  ledger itself.
- contact_permissions: append-only ALLOWED / OPTED_OUT / SUPPRESSED records per
  customer and channel, read by the Policy Engine.
- event_identity_aliases: identities under which existing events are recognised,
  written only by identity reconciliation. No existing row is rewritten.

All additive; existing rows upgrade in place.

Revision ID: e7b3c1d9f042
Revises: d4a7c91e5b20
Create Date: 2026-09-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e7b3c1d9f042'
down_revision: Union[str, Sequence[str], None] = 'd4a7c91e5b20'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('datasets', sa.Column('declared_domain', sa.String(), nullable=True))
    op.add_column('datasets', sa.Column('source_system', sa.String(), nullable=True))
    op.add_column('datasets', sa.Column('domain_classification', sa.JSON(), nullable=True))

    op.add_column('revenue_events', sa.Column('domain_basis', sa.String(), nullable=True))
    op.add_column('ingestion_runs', sa.Column('domain_classification', sa.JSON(), nullable=True))

    op.add_column(
        'execution_attempts', sa.Column('policy_decision_id', sa.String(), nullable=True)
    )
    op.add_column(
        'execution_attempts', sa.Column('contact_permission', sa.String(), nullable=True)
    )
    op.create_index(
        op.f('ix_execution_attempts_policy_decision_id'),
        'execution_attempts',
        ['policy_decision_id'],
        unique=False,
    )

    op.create_table(
        'contact_permissions',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('customer_id', sa.String(), nullable=False),
        sa.Column('channel', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('reason', sa.String(), nullable=True),
        sa.Column('effective_at', sa.DateTime(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_contact_permissions_customer_id'),
        'contact_permissions',
        ['customer_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_contact_permissions_effective_at'),
        'contact_permissions',
        ['effective_at'],
        unique=False,
    )

    op.create_table(
        'event_identity_aliases',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('event_id', sa.String(), nullable=False),
        sa.Column('alias_system', sa.String(), nullable=False),
        sa.Column('alias_event_id', sa.String(), nullable=False),
        sa.Column('method', sa.String(), nullable=False),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('reconciliation_run_id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['event_id'], ['revenue_events.event_id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('alias_system', 'alias_event_id', name='uq_event_identity_alias'),
    )
    op.create_index(
        op.f('ix_event_identity_aliases_event_id'),
        'event_identity_aliases',
        ['event_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_event_identity_aliases_reconciliation_run_id'),
        'event_identity_aliases',
        ['reconciliation_run_id'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f('ix_event_identity_aliases_reconciliation_run_id'),
        table_name='event_identity_aliases',
    )
    op.drop_index(op.f('ix_event_identity_aliases_event_id'), table_name='event_identity_aliases')
    op.drop_table('event_identity_aliases')

    op.drop_index(op.f('ix_contact_permissions_effective_at'), table_name='contact_permissions')
    op.drop_index(op.f('ix_contact_permissions_customer_id'), table_name='contact_permissions')
    op.drop_table('contact_permissions')

    op.drop_index(
        op.f('ix_execution_attempts_policy_decision_id'), table_name='execution_attempts'
    )
    op.drop_column('execution_attempts', 'contact_permission')
    op.drop_column('execution_attempts', 'policy_decision_id')

    op.drop_column('ingestion_runs', 'domain_classification')
    op.drop_column('revenue_events', 'domain_basis')

    op.drop_column('datasets', 'domain_classification')
    op.drop_column('datasets', 'source_system')
    op.drop_column('datasets', 'declared_domain')
