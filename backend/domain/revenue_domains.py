"""Canonical revenue-recovery domains.

``RiskCategory`` is the canonical domain enumeration. Its stored values are kept as
they are: they are persisted in recovery_cases / revenue_events, and renaming them
would need a data migration (and an enum-type change on PostgreSQL) for no semantic
gain. ``DOMAIN_SPECS`` gives each value its canonical domain name and the semantics
the rest of the system reasons with:

- what the obligation reference must identify (obligation identity),
- what counts as recovery when an outcome is verified,
- what a partial recovery means.

A case has exactly ONE primary domain. Events that relate to each other are linked
through correlation (customer + domain + reference), never by giving one event two
domains. An event whose domain cannot be established is not turned into a case at all:
see application/domain_classification.py.
"""

from dataclasses import dataclass

from domain.models import RiskCategory


class DomainBasis:
    """Why an event carries its domain. Stored on every event (revenue_events.domain_basis)
    so "why was this classified as a subscription failure?" has a recorded answer."""

    DECLARED_BY_SOURCE = "DECLARED_BY_SOURCE"  # the API request named the category
    DECLARED_FOR_DATASET = "DECLARED_FOR_DATASET"  # a person/API declared the file's domain
    CATEGORY_COLUMN_MAPPED = "CATEGORY_COLUMN_MAPPED"  # column confirmed as EVENT_CATEGORY
    CATEGORY_COLUMN_RECOGNISED = "CATEGORY_COLUMN_RECOGNISED"  # see domain_classification
    SCHEMA_EVIDENCE = "SCHEMA_EVIDENCE"  # columns that exist only for one kind of obligation
    PAYMENT_FAILURE_EVIDENCE = "PAYMENT_FAILURE_EVIDENCE"  # payment outcome / decline codes
    # Read-side label only (never written): an event stored before Phase 4, whose domain
    # came from the retired column-name keyword heuristic and whose basis was never kept.
    LEGACY_UNRECORDED = "LEGACY_UNRECORDED"


@dataclass(frozen=True)
class DomainSpec:
    category: RiskCategory
    canonical_name: str
    label: str
    # Obligation identity: an event belongs to the obligation
    # (customer_id, domain, reference_id). This says what reference_id must identify.
    obligation_reference: str
    # Verification: what the settlement source must confirm before money counts as
    # recovered. Sending a message or attempting a retry is an ACTION RESULT, never this.
    verification_basis: str
    recovered_means: str
    partial_recovery: str


_PARTIAL_CLOSES = (
    "The case closes PARTIALLY_RECOVERED with the recovered amount and the outstanding "
    "balance recorded. It is not reopened; a later event for the same obligation opens a "
    "new case that inherits the obligation's retry, contact and stop history."
)

DOMAIN_SPECS: dict[RiskCategory, DomainSpec] = {
    RiskCategory.FAILED_PAYMENT: DomainSpec(
        category=RiskCategory.FAILED_PAYMENT,
        canonical_name="PAYMENT_FAILURE",
        label="Failed payment",
        obligation_reference="the payment / transaction id of the failed charge",
        verification_basis="PAYMENT_SETTLED",
        recovered_means="the charge (or its retry) settled with the payment provider",
        partial_recovery="Rare (partial capture). " + _PARTIAL_CLOSES,
    ),
    RiskCategory.CHECKOUT_ABANDONMENT: DomainSpec(
        category=RiskCategory.CHECKOUT_ABANDONMENT,
        canonical_name="CHECKOUT_ABANDONMENT",
        label="Abandoned checkout",
        obligation_reference="the checkout / cart / order id",
        verification_basis="CHECKOUT_CONVERTED",
        recovered_means="the abandoned order was completed and paid",
        partial_recovery="A partly paid order. " + _PARTIAL_CLOSES,
    ),
    RiskCategory.FAILED_SUBSCRIPTION: DomainSpec(
        category=RiskCategory.FAILED_SUBSCRIPTION,
        canonical_name="SUBSCRIPTION_FAILURE",
        label="Failed subscription",
        obligation_reference="the subscription billing attempt / subscription id",
        verification_basis="BILLING_RECOVERED",
        recovered_means="the failed renewal charge was collected",
        partial_recovery="A partly collected renewal. " + _PARTIAL_CLOSES,
    ),
    RiskCategory.OVERDUE_INVOICE: DomainSpec(
        category=RiskCategory.OVERDUE_INVOICE,
        canonical_name="INVOICE_OVERDUE",
        label="Overdue invoice",
        obligation_reference="the invoice number",
        verification_basis="INVOICE_SETTLED",
        recovered_means="payment was received against the invoice",
        partial_recovery="Common (part-payment of an invoice). " + _PARTIAL_CLOSES,
    ),
    RiskCategory.BROKEN_PROMISE: DomainSpec(
        category=RiskCategory.BROKEN_PROMISE,
        canonical_name="PROMISE_TO_PAY",
        label="Broken promise to pay",
        obligation_reference="the promise / arrangement id, or the obligation it covers",
        verification_basis="PROMISED_PAYMENT_RECEIVED",
        recovered_means="money arrived against the obligation - a new promise is not recovery",
        partial_recovery="A partly honoured promise. " + _PARTIAL_CLOSES,
    ),
}


def domain_spec(category) -> DomainSpec:
    return DOMAIN_SPECS[RiskCategory(category)]


class OutcomeKind:
    """The closed set of things an authoritative outcome event can mean. Never money
    unless SETTLEMENT (application/outcome_events.py enforces this, not the source's own
    label)."""

    SETTLEMENT = "SETTLEMENT"  # money received against the obligation
    CANCELLATION = "CANCELLATION"  # the obligation was withdrawn at the source
    PROMISE_CREATED = "PROMISE_CREATED"  # never money
    PROMISE_BROKEN = "PROMISE_BROKEN"  # never money


# The source vocabulary for outcome events, per domain: exact normalised values only
# (matched the same way as application/domain_classification.py matches category
# values - whole tokens, never substrings), so an outcome_type from another domain's
# vocabulary is refused rather than silently accepted (domain applicability, like
# actions in domain.models.ACTION_SPECS). "settled" vs "partially settled" is
# deliberately not a separate value: whether a settlement is partial or full is decided
# by comparing its AMOUNT to the obligation's outstanding balance, never by a label.
OUTCOME_TYPE_SPECS: dict[str, tuple[str, RiskCategory]] = {
    "PAYMENT_SETTLED": (OutcomeKind.SETTLEMENT, RiskCategory.FAILED_PAYMENT),
    "CHECKOUT_CONVERTED": (OutcomeKind.SETTLEMENT, RiskCategory.CHECKOUT_ABANDONMENT),
    "ORDER_CANCELLED": (OutcomeKind.CANCELLATION, RiskCategory.CHECKOUT_ABANDONMENT),
    "RENEWAL_RECOVERED": (OutcomeKind.SETTLEMENT, RiskCategory.FAILED_SUBSCRIPTION),
    "SUBSCRIPTION_CANCELLED": (OutcomeKind.CANCELLATION, RiskCategory.FAILED_SUBSCRIPTION),
    "INVOICE_PAYMENT_RECEIVED": (OutcomeKind.SETTLEMENT, RiskCategory.OVERDUE_INVOICE),
    "INVOICE_SETTLED": (OutcomeKind.SETTLEMENT, RiskCategory.OVERDUE_INVOICE),
    "INVOICE_PARTIALLY_SETTLED": (OutcomeKind.SETTLEMENT, RiskCategory.OVERDUE_INVOICE),
    "INVOICE_VOIDED": (OutcomeKind.CANCELLATION, RiskCategory.OVERDUE_INVOICE),
    "PROMISE_CREATED": (OutcomeKind.PROMISE_CREATED, RiskCategory.BROKEN_PROMISE),
    "PROMISE_BROKEN": (OutcomeKind.PROMISE_BROKEN, RiskCategory.BROKEN_PROMISE),
    "PROMISE_PAYMENT_RECEIVED": (OutcomeKind.SETTLEMENT, RiskCategory.BROKEN_PROMISE),
}


def outcome_type_spec(outcome_type: str) -> tuple[str, RiskCategory] | None:
    """(kind, domain) for a normalised outcome_type value, or None if unrecognised."""
    return OUTCOME_TYPE_SPECS.get(outcome_type)


def obligation_key(customer_id: str | None, category, reference_id: str | None) -> str | None:
    """The identity of the obligation an event belongs to, or None.

    Identity is (domain, customer, reference) for every domain; what the reference
    means differs per domain (DomainSpec.obligation_reference). There is deliberately
    no fallback:

    - never customer_id alone (a customer can owe several unrelated obligations),
    - never a synthetic reference (a positional "tx_<row>" once merged unrelated rows),
    - never across domains (a checkout and an invoice sharing an id are different).

    None means the obligation cannot be identified, so the event is never correlated
    with another one and forms its own case.
    """
    if not customer_id or not reference_id:
        return None
    return f"{RiskCategory(category).value}|{customer_id}|{reference_id}"
