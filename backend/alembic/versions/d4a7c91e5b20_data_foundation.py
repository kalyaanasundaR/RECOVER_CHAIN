"""data foundation: event provenance, ingestion runs, evaluation runs, dataset hash

Phase 3.

- revenue_events gains provenance (source_type, source_ref, ingestion_run_id,
  ingested_at) so every event can answer "where did this come from?". Previously a
  file-replayed event recorded only the string "Dataset-<id>" in external_system, and
  RevenueEvent.ingested_at was never persisted.
- ingestion_runs records each batch ingestion: rows received / accepted / rejected /
  duplicate / not processed, reasons, and a bounded sample of rejected rows. That
  result used to exist only in one HTTP response.
- evaluation_runs stores evaluation harness results. Evaluation scenarios now run in an
  isolated ephemeral store and never write to recovery_cases / revenue_events /
  audit_records, which previously let POST /evaluation/run change operational
  dashboard money figures.
- datasets.content_sha256 identifies byte-identical re-uploads.

All additive and nullable; existing rows upgrade in place (legacy events simply have
no provenance recorded).

Revision ID: d4a7c91e5b20
Revises: c8d5e0a41f27
Create Date: 2026-09-14

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd4a7c91e5b20'
down_revision: Union[str, Sequence[str], None] = 'c8d5e0a41f27'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('revenue_events', sa.Column('source_type', sa.String(), nullable=True))
    op.add_column('revenue_events', sa.Column('source_ref', sa.String(), nullable=True))
    op.add_column('revenue_events', sa.Column('ingestion_run_id', sa.String(), nullable=True))
    op.add_column('revenue_events', sa.Column('ingested_at', sa.DateTime(), nullable=True))
    op.create_index(
        op.f('ix_revenue_events_source_ref'), 'revenue_events', ['source_ref'], unique=False
    )
    op.create_index(
        op.f('ix_revenue_events_ingestion_run_id'),
        'revenue_events',
        ['ingestion_run_id'],
        unique=False,
    )

    op.create_table(
        'ingestion_runs',
        sa.Column('run_id', sa.String(), nullable=False),
        sa.Column('source_type', sa.String(), nullable=False),
        sa.Column('source_ref', sa.String(), nullable=True),
        sa.Column('source_name', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('rows_in_source', sa.Integer(), nullable=True),
        sa.Column('rows_received', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rows_accepted', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rows_rejected', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rows_duplicate', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rows_not_at_risk', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rows_not_processed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('cases_created', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('cases_updated', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rejection_counts', sa.JSON(), nullable=True),
        sa.Column('rejected_samples', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('run_id'),
    )
    op.create_index(
        op.f('ix_ingestion_runs_source_ref'), 'ingestion_runs', ['source_ref'], unique=False
    )
    op.create_index(
        op.f('ix_ingestion_runs_started_at'), 'ingestion_runs', ['started_at'], unique=False
    )

    op.create_table(
        'evaluation_runs',
        sa.Column('run_id', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('experiment_mode', sa.String(), nullable=False),
        sa.Column('isolation', sa.String(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('total_scenarios', sa.Integer(), nullable=True),
        sa.Column('passed_scenarios', sa.Integer(), nullable=True),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('run_id'),
    )
    op.create_index(
        op.f('ix_evaluation_runs_started_at'), 'evaluation_runs', ['started_at'], unique=False
    )

    op.add_column('datasets', sa.Column('content_sha256', sa.String(), nullable=True))
    op.create_index(
        op.f('ix_datasets_content_sha256'), 'datasets', ['content_sha256'], unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_datasets_content_sha256'), table_name='datasets')
    op.drop_column('datasets', 'content_sha256')

    op.drop_index(op.f('ix_evaluation_runs_started_at'), table_name='evaluation_runs')
    op.drop_table('evaluation_runs')

    op.drop_index(op.f('ix_ingestion_runs_started_at'), table_name='ingestion_runs')
    op.drop_index(op.f('ix_ingestion_runs_source_ref'), table_name='ingestion_runs')
    op.drop_table('ingestion_runs')

    op.drop_index(op.f('ix_revenue_events_ingestion_run_id'), table_name='revenue_events')
    op.drop_index(op.f('ix_revenue_events_source_ref'), table_name='revenue_events')
    op.drop_column('revenue_events', 'ingested_at')
    op.drop_column('revenue_events', 'ingestion_run_id')
    op.drop_column('revenue_events', 'source_ref')
    op.drop_column('revenue_events', 'source_type')
