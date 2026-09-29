"""Phase 8A/8B — the closed provenance vocabulary for an outcome event.

Every outcome_events row must say where it actually came from, so a test/synthetic
event can never silently be counted as authoritative production evidence (see
docs/history/PHASE_8A_DATA_READINESS_REPORT.md, Step 2/3, and
docs/history/PHASE_8B_INDEPENDENT_EVIDENCE_REPORT.md, Step 5).

CRITICAL: at ingestion time, an omitted or unrecognised provenance value is
NEVER defaulted to PRODUCTION. The only safe default is TEST.

Phase 8B closes a real gap Phase 8A left open: defining PRODUCTION/PAYMENT_PROVIDER/
etc. as *values* is not the same as preventing a caller from simply typing one. Any
caller holding the shared dev API key could previously POST /outcomes with
provenance="production" and have it accepted at face value - there was no mechanism
verifying the claim. ACCEPTED_VIA_SHARED_API_KEY below is the actual enforcement: only
the provenance values this deployment can currently back are accepted through the
ordinary authenticated endpoint. A value that legitimately requires a stronger
mechanism (a registered source, a signed webhook, a dedicated integration credential -
none of which exist in this codebase) is rejected with a clear reason, not silently
accepted on the caller's word.
"""

from __future__ import annotations


class OutcomeProvenance:
    """Where an outcome event was actually obtained. A closed vocabulary - an
    unrecognised string is rejected, not silently accepted as a new category."""

    # A real integration reporting a real settlement/cancellation from a live
    # merchant system. Nothing in this codebase can produce this today - there is
    # no live payment gateway, invoicing system, or subscription billing connector
    # (application/execution engine is MockExecutionAdapter only; see the Phase 5/6
    # reports). Reserved for when a real integration AND a real authentication
    # mechanism for it (source registration, signed webhook, dedicated credential)
    # exist. Not accepted via the shared API key - see ACCEPTED_VIA_SHARED_API_KEY.
    PRODUCTION = "production"

    # A merchant operator (or any authenticated human) manually confirming what
    # actually happened - "yes, this customer really paid" / "no, they didn't".
    # This IS legitimate real-world evidence even without a live integration, and is
    # the most realistic near-term source of genuine outcome labels this system can
    # collect today. The shared API key is the only authentication this deployment
    # has for "an authenticated human/operator" - accepted on that basis.
    HUMAN_CONFIRMED = "human_confirmed"

    # Per-domain external system categories the brief names explicitly. Reserved for
    # when a real connector AND its own authentication mechanism exist (same
    # reasoning as PRODUCTION). Not accepted via the shared API key.
    PAYMENT_PROVIDER = "payment_provider"
    INVOICE_SYSTEM = "invoice_system"
    SUBSCRIPTION_SYSTEM = "subscription_system"
    CHECKOUT_SYSTEM = "checkout_system"

    # Phase 8B: clearly-labelled data shaped like a real external source's export/
    # webhook (a payment-provider settlement file, a checkout-conversion feed, ...)
    # but deliberately generated for this system, never claiming to be real. Exists
    # to prove an independent-source ingestion mechanism works, nothing more. Must
    # NEVER be counted as production evidence - see REAL_WORLD_VALUES below, which
    # excludes it exactly like TEST.
    EXTERNAL_SOURCE_SIMULATED = "external_source_simulated"

    # Everything produced by a test, a demo, a dataset-validation run, a runtime
    # verification pass, or any other non-real source. This is the ONLY safe default.
    TEST = "test"


# The full closed vocabulary a stored row may hold.
ALL_VALUES = frozenset(
    {
        OutcomeProvenance.PRODUCTION,
        OutcomeProvenance.HUMAN_CONFIRMED,
        OutcomeProvenance.PAYMENT_PROVIDER,
        OutcomeProvenance.INVOICE_SYSTEM,
        OutcomeProvenance.SUBSCRIPTION_SYSTEM,
        OutcomeProvenance.CHECKOUT_SYSTEM,
        OutcomeProvenance.EXTERNAL_SOURCE_SIMULATED,
        OutcomeProvenance.TEST,
    }
)

# Provenance values this deployment can ACTUALLY back with a real, live source right
# now. Empty today - no live payment gateway, invoicing, subscription, or checkout
# integration exists (Phase 5/6: "MockExecutionAdapter ... live gateways: false").
LIVE_INTEGRATION_BACKED: frozenset[str] = frozenset()

# The provenance values that count as genuinely authoritative, real-world evidence -
# as opposed to TEST or EXTERNAL_SOURCE_SIMULATED, neither of which ever does. A
# future automated integration would fall under PAYMENT_PROVIDER/INVOICE_SYSTEM/
# SUBSCRIPTION_SYSTEM/CHECKOUT_SYSTEM/PRODUCTION; a human manually confirming an
# outcome today is HUMAN_CONFIRMED.
REAL_WORLD_VALUES: frozenset[str] = frozenset(
    ALL_VALUES - {OutcomeProvenance.TEST, OutcomeProvenance.EXTERNAL_SOURCE_SIMULATED}
)

# Phase 8B/9: values accepted with NO further evidence beyond the shared API key
# itself - TEST needs nothing, HUMAN_CONFIRMED needs a non-RecoverChain-generated
# evidence_reference (application/outcome_events.py enforces this at ingestion),
# EXTERNAL_SOURCE_SIMULATED is honest about being simulated by construction.
NO_SOURCE_NEEDED: frozenset[str] = frozenset(
    {
        OutcomeProvenance.TEST,
        OutcomeProvenance.HUMAN_CONFIRMED,
        OutcomeProvenance.EXTERNAL_SOURCE_SIMULATED,
    }
)

# Phase 9: values that require a REGISTERED, VERIFIED, ENABLED evidence source of the
# matching type/domain (application/evidence_sources.py) - never accepted on the
# caller's bare word. In this environment none can currently be VERIFIED (no real
# provider credentials exist - see evidence_sources.CREDENTIAL_ENV_VARS), so these
# remain unreachable for a real, checkable reason, not a hardcoded blanket rejection.
SOURCE_GATED_VALUES: frozenset[str] = frozenset(
    {
        OutcomeProvenance.PRODUCTION,
        OutcomeProvenance.PAYMENT_PROVIDER,
        OutcomeProvenance.INVOICE_SYSTEM,
        OutcomeProvenance.SUBSCRIPTION_SYSTEM,
        OutcomeProvenance.CHECKOUT_SYSTEM,
    }
)

# Backward-compatible name (Phase 8B). Historically meant "what the shared API key may
# accept" - now exactly NO_SOURCE_NEEDED, since the source-gated values are no longer a
# blanket rejection but a real, checkable authorization (see SOURCE_GATED_VALUES and
# application/outcome_events.py's source-authorization step).
ACCEPTED_VIA_SHARED_API_KEY = NO_SOURCE_NEEDED


def normalize_provenance(value: str | None) -> str:
    """A validated provenance value, or TEST if none was given. Raises ValueError for
    an unrecognised value - never silently coerced to something it didn't say. This
    function only validates the VOCABULARY; whether a SOURCE_GATED_VALUES value is
    actually authorized (a verified, enabled, matching evidence source) is checked
    separately, in application/outcome_events.py, because it requires a database
    lookup this pure function deliberately does not perform."""
    if value is None or not str(value).strip():
        return OutcomeProvenance.TEST
    normalised = str(value).strip().lower()
    if normalised not in ALL_VALUES:
        raise ValueError(f"{value!r} is not a recognised outcome provenance")
    return normalised
