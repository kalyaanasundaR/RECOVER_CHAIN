"""outcome lifecycle: outcome events, promises, customer identity links, case position

Phase 5.

- recovery_cases gains recovered_amount, outstanding_amount, cancelled_amount: the
  case's financial position, maintained only by application.outcome_events from the
  append-only outcome_events table (never by reprocessing the pipeline). Backfilled for
  existing rows: outstanding_amount = amount_at_risk, recovered_amount = 0,
  cancelled_amount = 0 (their prior simulated 'outcome' JSON, if any, is left in place
  but is no longer authoritative - see the Phase 5 report).
- CaseState / RecoveryOutcomeStatus gain CANCELLED. Stored as a plain String column, so
  no PostgreSQL enum-type migration is needed here (see infrastructure/orm.py).
- outcome_events: one row per authoritative outcome event (settlement, cancellation,
  promise created/broken), append-only, with its match and reconciliation result.
- promises: one row per PROMISE_CREATED outcome event.
- customer_identity_links: deterministic (system, external id) -> customer_id mapping,
  written only by an explicit call - never inferred.
- contact_permissions gains source_record_id, ingestion_run_id, content_hash (unique
  with customer_id) for Phase 5 consent import (application/contact_permissions.py).

All additive; existing rows upgrade in place.

Revision ID: f3a9c7d1b284
Revises: e7b3c1d9f042
Create Date: 2026-09-15

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3a9c7d1b284"
down_revision: Union[str, Sequence[str], None] = "e7b3c1d9f042"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "recovery_cases",
        sa.Column(
            "recovered_amount",
            sa.Numeric(precision=18, scale=4),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "recovery_cases",
        sa.Column("outstanding_amount", sa.Numeric(precision=18, scale=4), nullable=True),
    )
    op.add_column(
        "recovery_cases",
        sa.Column(
            "cancelled_amount",
            sa.Numeric(precision=18, scale=4),
            nullable=False,
            server_default="0",
        ),
    )
    # Backfill; the column stays nullable (see infrastructure/orm.py) - a NULL is read
    # back as "not yet computed" and treated as the full amount still outstanding.
    op.execute("UPDATE recovery_cases SET outstanding_amount = amount_at_risk")

    op.add_column("contact_permissions", sa.Column("source_record_id", sa.String(), nullable=True))
    op.add_column("contact_permissions", sa.Column("ingestion_run_id", sa.String(), nullable=True))
    op.add_column("contact_permissions", sa.Column("content_hash", sa.String(), nullable=True))
    op.create_index(
        op.f("ix_contact_permissions_ingestion_run_id"),
        "contact_permissions",
        ["ingestion_run_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_contact_permissions_content_hash"),
        "contact_permissions",
        ["content_hash"],
        unique=False,
    )
    # SQLite cannot ALTER a table to add a constraint outside batch mode.
    with op.batch_alter_table("contact_permissions") as batch_op:
        batch_op.create_unique_constraint(
            "uq_contact_permission_content", ["customer_id", "content_hash"]
        )

    op.create_table(
        "outcome_events",
        sa.Column("outcome_event_id", sa.String(), nullable=False),
        sa.Column("external_system", sa.String(), nullable=False),
        sa.Column("external_event_id", sa.String(), nullable=False),
        sa.Column("domain", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("outcome_type", sa.String(), nullable=False),
        sa.Column("customer_id", sa.String(), nullable=False),
        sa.Column("customer_system", sa.String(), nullable=True),
        sa.Column("reference_id", sa.String(), nullable=True),
        sa.Column("amount", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("currency", sa.String(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(), nullable=False),
        sa.Column("promised_date", sa.DateTime(), nullable=True),
        sa.Column("raw_payload", sa.JSON(), nullable=True),
        sa.Column("source_type", sa.String(), nullable=True),
        sa.Column("source_ref", sa.String(), nullable=True),
        sa.Column("ingestion_run_id", sa.String(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(), nullable=False),
        sa.Column("match_status", sa.String(), nullable=False),
        sa.Column("match_reason", sa.String(), nullable=True),
        sa.Column("case_id", sa.String(), nullable=True),
        sa.Column("applied_amount", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("excess_amount", sa.Numeric(precision=18, scale=4), nullable=True),
        sa.Column("effect", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["case_id"], ["recovery_cases.case_id"]),
        sa.PrimaryKeyConstraint("outcome_event_id"),
        sa.UniqueConstraint(
            "external_system", "external_event_id", name="uq_outcome_event_external_id"
        ),
    )
    op.create_index(
        op.f("ix_outcome_events_customer_id"), "outcome_events", ["customer_id"], unique=False
    )
    op.create_index(
        op.f("ix_outcome_events_reference_id"), "outcome_events", ["reference_id"], unique=False
    )
    op.create_index(
        op.f("ix_outcome_events_source_ref"), "outcome_events", ["source_ref"], unique=False
    )
    op.create_index(
        op.f("ix_outcome_events_ingestion_run_id"),
        "outcome_events",
        ["ingestion_run_id"],
        unique=False,
    )
    op.create_index(op.f("ix_outcome_events_case_id"), "outcome_events", ["case_id"], unique=False)

    op.create_table(
        "promises",
        sa.Column("promise_id", sa.String(), nullable=False),
        sa.Column("outcome_event_id", sa.String(), nullable=False),
        sa.Column("case_id", sa.String(), nullable=False),
        sa.Column("domain", sa.String(), nullable=False),
        sa.Column("customer_id", sa.String(), nullable=False),
        sa.Column("reference_id", sa.String(), nullable=True),
        sa.Column("promised_amount", sa.Numeric(precision=18, scale=4), nullable=False),
        sa.Column("currency", sa.String(), nullable=False),
        sa.Column("promised_date", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("broken_reported_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["outcome_event_id"], ["outcome_events.outcome_event_id"]),
        sa.ForeignKeyConstraint(["case_id"], ["recovery_cases.case_id"]),
        sa.PrimaryKeyConstraint("promise_id"),
        sa.UniqueConstraint("outcome_event_id"),
    )
    op.create_index(op.f("ix_promises_case_id"), "promises", ["case_id"], unique=False)
    op.create_index(op.f("ix_promises_customer_id"), "promises", ["customer_id"], unique=False)

    op.create_table(
        "customer_identity_links",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("system", sa.String(), nullable=False),
        sa.Column("external_customer_id", sa.String(), nullable=False),
        sa.Column("customer_id", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("system", "external_customer_id", name="uq_customer_identity_link"),
    )
    op.create_index(
        op.f("ix_customer_identity_links_customer_id"),
        "customer_identity_links",
        ["customer_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_customer_identity_links_customer_id"), table_name="customer_identity_links"
    )
    op.drop_table("customer_identity_links")

    op.drop_index(op.f("ix_promises_customer_id"), table_name="promises")
    op.drop_index(op.f("ix_promises_case_id"), table_name="promises")
    op.drop_table("promises")

    op.drop_index(op.f("ix_outcome_events_case_id"), table_name="outcome_events")
    op.drop_index(op.f("ix_outcome_events_ingestion_run_id"), table_name="outcome_events")
    op.drop_index(op.f("ix_outcome_events_source_ref"), table_name="outcome_events")
    op.drop_index(op.f("ix_outcome_events_reference_id"), table_name="outcome_events")
    op.drop_index(op.f("ix_outcome_events_customer_id"), table_name="outcome_events")
    op.drop_table("outcome_events")

    with op.batch_alter_table("contact_permissions") as batch_op:
        batch_op.drop_constraint("uq_contact_permission_content", type_="unique")
    op.drop_index(op.f("ix_contact_permissions_content_hash"), table_name="contact_permissions")
    op.drop_index(op.f("ix_contact_permissions_ingestion_run_id"), table_name="contact_permissions")
    op.drop_column("contact_permissions", "content_hash")
    op.drop_column("contact_permissions", "ingestion_run_id")
    op.drop_column("contact_permissions", "source_record_id")

    op.drop_column("recovery_cases", "cancelled_amount")
    op.drop_column("recovery_cases", "outstanding_amount")
    op.drop_column("recovery_cases", "recovered_amount")
