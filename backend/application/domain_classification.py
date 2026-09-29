"""Deterministic domain classification for file sources.

An API event names its domain (``risk_category`` is a required field). A file row does
not, so the domain of a dataset is established here, in this order:

1. DECLARED  - a domain declared for the whole dataset (PUT /datasets/{id}/domain, or
               ``declared_domain`` on mapping confirmation). Used as given.
2. DECLARED  - a column the user confirmed as ``EVENT_CATEGORY``. Each row's value must
               be one of the exact domain names below; a row whose value is not is
               rejected (``unresolved_domain``), never defaulted.
3. INFERRED  - a recognised category column (exact name) in which EVERY value names a
               domain: classified row by row.
4. INFERRED  - columns that exist only for one kind of obligation (``billing_cycle``,
               ``aging_bucket``, ``checkout_phase`` ...), for exactly one domain.
5. INFERRED  - payment-failure evidence (a payment outcome column, or a reason column
               whose values are mostly payment decline codes) and nothing else: a
               standalone failed payment.
6. NEEDS_MAPPING - anything else, including conflicting evidence. Nothing is guessed:
               the dataset cannot generate cases until its domain is declared.

Every rule matches whole, normalised column names or values from explicit lists
(``invoiceNumber``, ``Invoice Number`` and ``invoice-number`` all normalise to
``invoice_number``). No rule tests whether a word merely appears inside a name, so an
incidental "subscription" in an unrelated column cannot change a row's domain, and
column order has no effect. No model or LLM is involved.

Deliberately NOT evidence: ``invoice_date`` (used as a generic date column in the shipped
samples), ``days_overdue`` (invoices and subscription dunning both have it), ``status`` /
``result`` (every domain has an outcome), and payment-method columns (every domain is
paid somehow).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from application.ingestion import _is_missing
from domain.models import RiskCategory
from domain.revenue_domains import DOMAIN_SPECS, DomainBasis

CLASSIFIER_VERSION = "domain-classifier-v1"

DECLARED = "DECLARED"
INFERRED = "INFERRED"
NEEDS_MAPPING = "NEEDS_MAPPING"

MODE_DATASET = "DATASET"  # one domain for every row
MODE_PER_ROW = "PER_ROW"  # each row's domain comes from its category column

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_token(value: Any) -> str:
    """'invoiceNumber' / 'Invoice Number' / 'invoice-number' -> 'invoice_number'."""
    text = _CAMEL.sub("_", str(value).strip())
    return _NON_ALNUM.sub("_", text.lower()).strip("_")


# Values that name a domain in a category column. Whole normalised values only.
_VALUE_VOCABULARY: dict[RiskCategory, frozenset[str]] = {
    RiskCategory.FAILED_PAYMENT: frozenset(
        {"payment", "payments", "failed_payment", "payment_failed", "card_payment", "charge"}
    ),
    RiskCategory.CHECKOUT_ABANDONMENT: frozenset(
        {"checkout", "abandoned_checkout", "cart_abandonment", "abandoned_cart"}
    ),
    RiskCategory.FAILED_SUBSCRIPTION: frozenset(
        {
            "subscription",
            "failed_subscription",
            "subscription_renewal",
            "renewal",
            "failed_renewal",
            "recurring_billing",
        }
    ),
    RiskCategory.OVERDUE_INVOICE: frozenset(
        {"invoice", "invoices", "accounts_receivable", "receivable", "receivables"}
    ),
    RiskCategory.BROKEN_PROMISE: frozenset(
        {"promise", "broken_promise", "promise_to_pay", "ptp", "missed_promise"}
    ),
}


def _build_value_index() -> dict[str, RiskCategory]:
    index: dict[str, RiskCategory] = {}
    for domain, words in _VALUE_VOCABULARY.items():
        spec = DOMAIN_SPECS[domain]
        for word in {*words, normalize_token(domain.value), normalize_token(spec.canonical_name)}:
            if word in index and index[word] != domain:  # pragma: no cover - table error
                raise ValueError(f"domain value {word!r} is assigned to two domains")
            index[word] = domain
    return index


_VALUE_TO_DOMAIN = _build_value_index()

# Column names that hold a per-row domain.
CATEGORY_COLUMN_NAMES = frozenset(
    {
        "event_type",
        "event_category",
        "revenue_stream",
        "revenue_domain",
        "risk_category",
        "recovery_category",
        "domain",
        "category",
    }
)

# Columns that exist only for one kind of obligation.
DISTINCTIVE_COLUMNS: dict[RiskCategory, frozenset[str]] = {
    RiskCategory.CHECKOUT_ABANDONMENT: frozenset(
        {
            "cart_id",
            "cart_total",
            "cart_value",
            "checkout_id",
            "checkout_phase",
            "checkout_stage",
            "checkout_step",
            "basket_id",
            "basket_total",
        }
    ),
    RiskCategory.FAILED_SUBSCRIPTION: frozenset(
        {
            "subscription_id",
            "subscription_plan",
            "plan_id",
            "billing_cycle",
            "billing_period",
            "renewal_date",
            "mrr",
            "mrr_amount",
        }
    ),
    RiskCategory.OVERDUE_INVOICE: frozenset(
        {
            "invoice_id",
            "invoice_number",
            "invoice_no",
            "invoice_amount",
            "due_date",
            "aging_bucket",
            "days_past_due",
            "payment_terms",
        }
    ),
    RiskCategory.BROKEN_PROMISE: frozenset(
        {
            "promise_id",
            "promise_date",
            "promised_date",
            "promised_amount",
            "ptp_date",
            "ptp_amount",
        }
    ),
}

# Money going OUT (payouts, refunds) is not a revenue-recovery obligation.
UNSUPPORTED_OBLIGATION_COLUMNS = frozenset(
    {"payout_id", "payout_amount", "refund_id", "refund_amount"}
)

PAYMENT_OUTCOME_COLUMNS = frozenset(
    {
        "payment_status",
        "payment_result",
        "payment_outcome",
        "payment_failed",
        "paid_status",
        "charge_status",
        "charge_result",
    }
)
FAILURE_CODE_COLUMNS = frozenset(
    {
        "failure_code",
        "failure_reason",
        "decline_code",
        "decline_reason",
        "error_code",
        "reason",
        "reason_code",
        "return_code",
        "return_reason",
    }
)
PAYMENT_FAILURE_CODES = frozenset(
    {
        "insufficient_funds",
        "nsf",
        "card_declined",
        "declined",
        "generic_decline",
        "do_not_honor",
        "expired_card",
        "card_expired",
        "invalid_expiry",
        "incorrect_cvc",
        "invalid_cvc",
        "incorrect_number",
        "lost_card",
        "stolen_card",
        "pickup_card",
        "restricted_card",
        "fraudulent",
        "card_velocity_exceeded",
        "withdrawal_count_limit_exceeded",
        "processing_error",
        "network_error",
        "gateway_timeout",
        "timeout",
        "processor_unavailable",
        "try_again_later",
        "authentication_required",
        "3ds_failed",
        "bank_rejected",
        "account_closed",
        "invalid_account_number",
        "no_account",
        "currency_not_supported",
        "mandate_cancelled",
        "revoked",
    }
)
# A reason column counts as payment evidence only when most of its values are decline
# codes; a few "insufficient_funds" among invoice notes are not.
PAYMENT_CODE_MIN_SHARE = 0.5

_MAX_LISTED = 20


def domain_for_value(value: Any) -> RiskCategory | None:
    """The domain a category-column value names, or None. Exact normalised match."""
    if _is_missing(value):
        return None
    return _VALUE_TO_DOMAIN.get(normalize_token(value))


@dataclass
class DomainClassification:
    status: str  # DECLARED | INFERRED | NEEDS_MAPPING
    reason: str
    mode: str | None = None  # DATASET | PER_ROW
    domain: RiskCategory | None = None  # for MODE_DATASET
    basis: str | None = None  # DomainBasis for the rows it classifies
    category_column: str | None = None  # for MODE_PER_ROW
    value_domains: dict[str, str] = field(default_factory=dict)  # PER_ROW value -> domain
    unresolved_values: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    inferred: dict | None = None  # what inference found, kept when a domain was declared
    rows_examined: int = 0

    @property
    def needs_mapping(self) -> bool:
        return self.status == NEEDS_MAPPING

    def resolve(self, row: dict) -> tuple[RiskCategory, str] | None:
        """(domain, basis) for one row, or None when the row's domain is unresolved."""
        if self.status == NEEDS_MAPPING:
            return None
        if self.mode == MODE_DATASET:
            return self.domain, self.basis
        domain = domain_for_value(row.get(self.category_column))
        return (domain, self.basis) if domain is not None else None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "mode": self.mode,
            "domain": self.domain.value if self.domain else None,
            "canonical_domain": DOMAIN_SPECS[self.domain].canonical_name if self.domain else None,
            "basis": self.basis,
            "category_column": self.category_column,
            "value_domains": self.value_domains,
            "unresolved_values": self.unresolved_values,
            "reason": self.reason,
            "evidence": self.evidence,
            "warnings": self.warnings,
            "inferred": self.inferred,
            "rows_examined": self.rows_examined,
            "classifier_version": CLASSIFIER_VERSION,
        }


def _column_values(df, column) -> list:
    return [v for v in df[column].tolist() if not _is_missing(v)]


def _category_stats(values: list) -> tuple[dict[str, str], list[str]]:
    distinct = sorted({normalize_token(v) for v in values})
    mapping = {v: _VALUE_TO_DOMAIN[v].value for v in distinct if v in _VALUE_TO_DOMAIN}
    unresolved = [v for v in distinct if v not in _VALUE_TO_DOMAIN]
    return mapping, unresolved


def _infer(df, columns: list[str], norm: dict[str, str]) -> DomainClassification:
    evidence: list[dict] = []
    rows = len(df)

    # 3. A recognised category column in which every value names a domain.
    for col in columns:
        if norm[col] not in CATEGORY_COLUMN_NAMES:
            continue
        values = _column_values(df, col)
        mapping, unresolved = _category_stats(values)
        if values and not unresolved:
            return DomainClassification(
                status=INFERRED,
                mode=MODE_PER_ROW,
                basis=DomainBasis.CATEGORY_COLUMN_RECOGNISED,
                category_column=col,
                value_domains=mapping,
                reason=(
                    f"Column '{col}' names a domain on every row: "
                    + ", ".join(f"{v} -> {d}" for v, d in mapping.items())
                ),
                evidence=[{"kind": "category_column", "column": col, "values": mapping}],
                rows_examined=rows,
            )
        evidence.append(
            {
                "kind": "category_column_not_usable",
                "column": col,
                "unresolved_values": unresolved[:_MAX_LISTED],
            }
        )

    unsupported = [c for c in columns if norm[c] in UNSUPPORTED_OBLIGATION_COLUMNS]
    if unsupported:
        return DomainClassification(
            status=NEEDS_MAPPING,
            reason=(
                f"Columns {unsupported} describe money paid out or refunded, which is not a "
                "revenue-recovery obligation. Declare the domain if this file is one."
            ),
            evidence=evidence + [{"kind": "unsupported_obligation", "columns": unsupported}],
            rows_examined=rows,
        )

    # 4. Columns that exist only for one kind of obligation.
    hits: dict[RiskCategory, list[str]] = {}
    for col in columns:
        for domain, names in DISTINCTIVE_COLUMNS.items():
            if norm[col] in names:
                hits.setdefault(domain, []).append(col)
    schema_evidence = [
        {"kind": "obligation_columns", "domain": d.value, "columns": cols}
        for d, cols in hits.items()
    ]
    if len(hits) == 1:
        ((domain, cols),) = hits.items()
        return DomainClassification(
            status=INFERRED,
            mode=MODE_DATASET,
            domain=domain,
            basis=DomainBasis.SCHEMA_EVIDENCE,
            reason=f"Columns {cols} exist only for {DOMAIN_SPECS[domain].label.lower()} data.",
            evidence=evidence + schema_evidence,
            rows_examined=rows,
        )
    if len(hits) > 1:
        return DomainClassification(
            status=NEEDS_MAPPING,
            reason="Conflicting evidence: "
            + "; ".join(f"{cols} suggest {d.value}" for d, cols in hits.items())
            + ". Declare the domain.",
            evidence=evidence + schema_evidence,
            rows_examined=rows,
        )

    # 5. Payment-failure evidence and nothing else.
    payment_evidence: list[dict] = []
    for col in columns:
        if norm[col] in PAYMENT_OUTCOME_COLUMNS:
            payment_evidence.append({"kind": "payment_outcome_column", "column": col})
        elif norm[col] in FAILURE_CODE_COLUMNS:
            values = _column_values(df, col)
            if not values:
                continue
            codes = [normalize_token(v) for v in values]
            share = sum(1 for c in codes if c in PAYMENT_FAILURE_CODES) / len(codes)
            if share >= PAYMENT_CODE_MIN_SHARE:
                payment_evidence.append(
                    {
                        "kind": "payment_decline_codes",
                        "column": col,
                        "share": round(share, 3),
                        "codes": sorted({c for c in codes if c in PAYMENT_FAILURE_CODES})[
                            :_MAX_LISTED
                        ],
                    }
                )
    if payment_evidence:
        cols = [e["column"] for e in payment_evidence]
        return DomainClassification(
            status=INFERRED,
            mode=MODE_DATASET,
            domain=RiskCategory.FAILED_PAYMENT,
            basis=DomainBasis.PAYMENT_FAILURE_EVIDENCE,
            reason=(
                f"Columns {cols} record payment failures and nothing identifies another "
                "kind of obligation."
            ),
            evidence=evidence + payment_evidence,
            rows_examined=rows,
        )

    return DomainClassification(
        status=NEEDS_MAPPING,
        reason=(
            "No declared domain, no recognised category column and no column that identifies "
            "the kind of obligation. Declare the domain - it is not assumed to be a failed "
            "payment."
        ),
        evidence=evidence,
        rows_examined=rows,
    )


def classify_dataset(
    df,
    *,
    declared_domain: str | None = None,
    category_column: str | None = None,
) -> DomainClassification:
    """Classify a dataset (a pandas DataFrame of its rows, or a sample of them)."""
    columns = [str(c) for c in df.columns]
    norm = {c: normalize_token(c) for c in columns}
    inferred = _infer(df, columns, norm)
    summary = {
        "status": inferred.status,
        "mode": inferred.mode,
        "domain": inferred.domain.value if inferred.domain else None,
        "category_column": inferred.category_column,
        "basis": inferred.basis,
        "reason": inferred.reason,
    }

    if declared_domain:
        domain = RiskCategory(declared_domain)
        warnings = []
        if inferred.status == INFERRED and (
            inferred.mode == MODE_PER_ROW or inferred.domain != domain
        ):
            warnings.append(
                f"The data suggests otherwise ({inferred.reason}); the declared domain "
                f"{domain.value} is applied to every row."
            )
        return DomainClassification(
            status=DECLARED,
            mode=MODE_DATASET,
            domain=domain,
            basis=DomainBasis.DECLARED_FOR_DATASET,
            reason=f"Declared for this dataset: {domain.value}.",
            evidence=[{"kind": "declared_domain", "domain": domain.value}],
            warnings=warnings,
            inferred=summary,
            rows_examined=len(df),
        )

    if category_column:
        if category_column not in columns:
            return DomainClassification(
                status=NEEDS_MAPPING,
                reason=f"The column mapped as EVENT_CATEGORY ('{category_column}') is not in the file.",
                inferred=summary,
                rows_examined=len(df),
            )
        mapping, unresolved = _category_stats(_column_values(df, category_column))
        warnings = []
        if unresolved:
            warnings.append(
                f"Rows whose '{category_column}' value is not a known domain are rejected: "
                f"{unresolved[:_MAX_LISTED]}"
            )
        return DomainClassification(
            status=DECLARED,
            mode=MODE_PER_ROW,
            basis=DomainBasis.CATEGORY_COLUMN_MAPPED,
            category_column=category_column,
            value_domains=mapping,
            unresolved_values=unresolved[:_MAX_LISTED],
            reason=f"Column '{category_column}' was confirmed as the per-row domain.",
            evidence=[{"kind": "category_column", "column": category_column, "values": mapping}],
            warnings=warnings,
            inferred=summary,
            rows_examined=len(df),
        )

    return inferred
