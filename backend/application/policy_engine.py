import os
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from application.trace_ids import trace_id_for_case
from domain.lifecycle import OUTCOME_FINAL_STATES
from domain.models import (
    ActionChannel,
    ActionType,
    PolicyDecision,
    PolicyDecisionStatus,
    RecoveryCase,
    RiskCategory,
    RuleResult,
    action_applies_to,
    action_channel,
)

UNKNOWN_CONSENT_BEHAVIOURS = ("ALLOW", "ESCALATE", "DENY")


def _unknown_consent_default(channel: str) -> str:
    return os.getenv(f"RECOVERCHAIN_UNKNOWN_CONSENT_{channel.upper()}", "ALLOW")


class MerchantPolicy(BaseModel):
    model_config = ConfigDict(validate_default=True)

    version: str = "merchant-policy-v1.1"
    payment_max_retries: int = 3
    payment_retry_cooldown_hours: int = 24
    communication_max_messages_24h: int = 2
    financial_max_automated_amount: float = 5000.0

    # What to do when a customer has NO contact-permission record for the channel.
    #
    # PRODUCT DECISION (Phase 4, stated here instead of implied): the default is ALLOW,
    # which keeps the behaviour RecoverChain has always had. UNKNOWN is still never
    # treated as consent - the decision records the permission as UNKNOWN, its rule text
    # says the action was allowed by this setting and not by consent, and the execution
    # ledger stores UNKNOWN against the attempt. A deployment that must not act without a
    # recorded permission sets ESCALATE (a person decides) or DENY, here or through
    # RECOVERCHAIN_UNKNOWN_CONSENT_COMMUNICATION / RECOVERCHAIN_UNKNOWN_CONSENT_PAYMENT.
    # Recorded OPTED_OUT and SUPPRESSED permissions are enforced whatever this says.
    unknown_consent_communication: str = Field(
        default_factory=lambda: _unknown_consent_default("communication")
    )
    unknown_consent_payment: str = Field(
        default_factory=lambda: _unknown_consent_default("payment")
    )

    @field_validator("unknown_consent_communication", "unknown_consent_payment")
    @classmethod
    def _unknown_consent_behaviour(cls, v: str) -> str:
        v = str(v).strip().upper()
        if v not in UNKNOWN_CONSENT_BEHAVIOURS:
            raise ValueError(f"must be one of {UNKNOWN_CONSENT_BEHAVIOURS}")
        return v


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


class PolicyContext(BaseModel):
    """Durable state for a policy decision, built by
    SqlAlchemyCaseRepository.get_policy_context from the execution_attempts ledger, the
    case lifecycle, the audit trail and the contact_permissions table. When supplied,
    the Policy Engine uses it instead of the legacy proxy on `case.linked_events` (which
    counted the triggering failure itself as a prior attempt).

    Unknown fields are rejected: `consent_ok` (always True, never populated) was removed
    in Phase 4, and silently ignoring a caller that still passed it would hide a missing
    consent check."""

    model_config = ConfigDict(extra="forbid")

    # Payment attempts for this OBLIGATION (sibling cases included).
    prior_payment_attempts: int = 0
    last_payment_attempt_at: datetime | None = None
    # Communication attempts to this CUSTOMER, across all of their cases, in the last 24h.
    comms_in_last_24h: int = 0
    stopped: bool = False

    # Contact permission per action channel ("ALLOWED" / "OPTED_OUT" / "SUPPRESSED" /
    # "UNKNOWN"), resolved by application.contact_permissions. UNKNOWN means no record
    # exists and is NOT consent: MerchantPolicy.unknown_consent_* decides what it allows.
    communication_permission: str = "UNKNOWN"
    payment_permission: str = "UNKNOWN"
    permission_records: dict[str, dict] = Field(default_factory=dict)  # channel -> record

    # Set when a human has explicitly reviewed and approved THIS case's escalation.
    # It satisfies the escalation trigger (the automated-amount threshold) and
    # NOTHING else: retry caps, cooldowns, contact limits, consent and stop rules are
    # still enforced against the approved action. Human approval is therefore an
    # authorisation to exceed the automation limit, never a bypass of the Policy
    # Engine.
    human_approved: bool = False


# Domain-specific policy extension point.
#
# A rule receives (case, context, policy) and returns None (not applicable) or
# (RuleResult, status_if_failed). Rules run after the shared rules, only for cases of
# their domain, and can only make a decision STRICTER (precedence DENIED > ESCALATE >
# WAIT > PERMITTED) - a domain rule can never permit what a shared rule refused. No
# domain has rules today: the audit found no current requirement that justifies
# different retry / contact / urgency semantics per domain, so every domain is held to
# the shared rules and payment behaviour is unchanged.
DomainPolicyRule = Callable[
    [RecoveryCase, "PolicyContext | None", MerchantPolicy],
    "tuple[RuleResult, PolicyDecisionStatus] | None",
]
DOMAIN_POLICY_RULES: dict[RiskCategory, tuple[DomainPolicyRule, ...]] = {
    category: () for category in RiskCategory
}

_PRECEDENCE = {
    PolicyDecisionStatus.PERMITTED: 0,
    PolicyDecisionStatus.WAIT: 1,
    PolicyDecisionStatus.ESCALATE: 2,
    PolicyDecisionStatus.DENIED: 3,
}


class DeterministicPolicyEngine:
    """
    Evaluates actions deterministically against a MerchantPolicy.
    Decision Precedence: DENIED > ESCALATE > WAIT > PERMITTED
    """

    def __init__(
        self,
        policy: MerchantPolicy | None = None,
        domain_rules: dict[RiskCategory, tuple[DomainPolicyRule, ...]] | None = None,
    ):
        self.policy = policy if policy is not None else MerchantPolicy()
        self.domain_rules = DOMAIN_POLICY_RULES if domain_rules is None else domain_rules

    def evaluate(self, case: RecoveryCase, context: PolicyContext | None = None) -> PolicyDecision:
        """Evaluate, then stamp the decision with its trace references (Phase 16). The
        stamp adds identifiers only - it never changes status, reason or any rule."""
        decision = self._evaluate(case, context)
        top = case.recommendation.top_candidate if case.recommendation else None
        decision.trace_id = trace_id_for_case(case.case_id)
        decision.case_id = case.case_id
        decision.action = top.action_type.value if top else None
        decision.reason_codes = [r.rule_name for r in decision.failed_rules] or ["ALL_RULES_PASSED"]
        decision.approval_required = decision.status == PolicyDecisionStatus.ESCALATE
        decision.lifecycle_state = getattr(case.current_state, "value", str(case.current_state))
        return decision

    def _evaluate(self, case: RecoveryCase, context: PolicyContext | None = None) -> PolicyDecision:
        rules_evaluated: list[RuleResult] = []
        failed_rules: list[RuleResult] = []

        status = PolicyDecisionStatus.PERMITTED
        reason = "All policy rules passed."

        # 0. Hard stops (only when a real context is supplied)
        if context is not None and context.stopped:
            return PolicyDecision(
                decision_id=f"pol_{uuid.uuid4().hex[:8]}",
                status=PolicyDecisionStatus.DENIED,
                policy_version=self.policy.version,
                rules_evaluated=[],
                failed_rules=[
                    RuleResult(
                        rule_name="StopRule",
                        passed=False,
                        details="Case is STOPPED; no further automated action.",
                    )
                ],
                reason="Case is STOPPED.",
            )

        # 0b. Obligation already settled or cancelled (Phase 5). A real outcome event
        # (application/outcome_events.py) is the only thing that can put a case here or
        # bring its outstanding_amount to zero; nothing about the recommendation or the
        # request influences this. Checked on the CASE itself (not PolicyContext) because
        # it is the case's own recorded financial position, not obligation history that
        # needs assembling from siblings. None (a hand-built case with no reconciled
        # position, e.g. most unit tests) is not evaluated - it means "unknown", not
        # "clear".
        if case.current_state in OUTCOME_FINAL_STATES or (
            case.outstanding_amount is not None and case.outstanding_amount.amount <= 0
        ):
            return PolicyDecision(
                decision_id=f"pol_{uuid.uuid4().hex[:8]}",
                status=PolicyDecisionStatus.DENIED,
                policy_version=self.policy.version,
                rules_evaluated=[],
                failed_rules=[
                    RuleResult(
                        rule_name="ObligationSettled",
                        passed=False,
                        details=(
                            f"This obligation is {case.current_state.value}; nothing "
                            "remains to recover."
                        ),
                    )
                ],
                reason="Obligation already settled or cancelled.",
            )

        # 1. Evidence Verification
        if not case.recommendation or not case.recommendation.top_candidate:
            return PolicyDecision(
                decision_id=f"pol_{uuid.uuid4().hex[:8]}",
                status=PolicyDecisionStatus.ESCALATE,
                policy_version=self.policy.version,
                rules_evaluated=[],
                failed_rules=[
                    RuleResult(
                        rule_name="EvidenceCheck",
                        passed=False,
                        details="No recommendation available.",
                    )
                ],
                reason="Insufficient evidence to authorize an action.",
            )

        action = case.recommendation.top_candidate.action_type

        # 2. Conflicting Diagnosis / Human Escalation Action
        if action == ActionType.ESCALATE_TO_HUMAN:
            r = RuleResult(
                rule_name="HumanEscalationAction",
                passed=False,
                details="Action explicitly requests escalation.",
            )
            rules_evaluated.append(r)
            failed_rules.append(r)
            status = PolicyDecisionStatus.ESCALATE
            reason = r.details

        # 3. Financial Limits
        if case.amount_at_risk.amount > self.policy.financial_max_automated_amount:
            if context is not None and context.human_approved:
                # A human reviewed this exact escalation and authorised it. The
                # threshold is satisfied; every other rule below still runs.
                rules_evaluated.append(
                    RuleResult(
                        rule_name="FinancialAutomatedLimit",
                        passed=True,
                        details=(
                            f"Amount {case.amount_at_risk.amount} exceeds automated threshold "
                            f"{self.policy.financial_max_automated_amount}; authorised by human review."
                        ),
                    )
                )
            else:
                r = RuleResult(
                    rule_name="FinancialAutomatedLimit",
                    passed=False,
                    details=f"Amount {case.amount_at_risk.amount} exceeds automated threshold {self.policy.financial_max_automated_amount}.",
                )
                rules_evaluated.append(r)
                failed_rules.append(r)
                status = PolicyDecisionStatus.ESCALATE
                reason = r.details
        else:
            rules_evaluated.append(
                RuleResult(
                    rule_name="FinancialAutomatedLimit", passed=True, details="Amount within limit."
                )
            )

        now = datetime.now(UTC)
        channel = action_channel(action)

        # 3a. The action must be a recovery action for this case's domain
        # (domain.models.ACTION_SPECS). The recommender only proposes applicable actions;
        # this is the Policy Engine's own structural check, so a checkout reminder can
        # never be authorised for an invoice whatever produced the recommendation.
        category = case.risk_category.value
        if action_applies_to(action, case.risk_category):
            rules_evaluated.append(
                RuleResult(
                    rule_name="ActionDomainApplicability",
                    passed=True,
                    details=f"{action.value} is a {channel.value} action for {category}.",
                )
            )
        else:
            r = RuleResult(
                rule_name="ActionDomainApplicability",
                passed=False,
                details=f"{action.value} is not a recovery action for {category}.",
            )
            rules_evaluated.append(r)
            failed_rules.append(r)
            status = PolicyDecisionStatus.DENIED
            reason = r.details

        # 3b. Contact permission, from durable state. Internal actions reach nobody.
        contact_permission = None
        if context is not None and channel in (ActionChannel.PAYMENT, ActionChannel.COMMUNICATION):
            status, reason, contact_permission = self._contact_permission(
                channel, context, status, reason, rules_evaluated, failed_rules
            )

        # 4. Action-Specific Limits
        #
        # Rate limits are selected by the action's CHANNEL, not by matching on the
        # action's name. The previous `"REMINDER" in action.value` test left five
        # customer-facing actions (REQUEST_PAYMENT_METHOD_UPDATE, SEND_PAYMENT_LINK,
        # OFFER_CHECKOUT_ASSISTANCE, ESCALATE_COLLECTION, REQUEST_NEW_COMMITMENT)
        # matching NEITHER branch, so they carried no frequency limit at all and could
        # be executed without bound. `action_channel` defaults to COMMUNICATION, so an
        # action type added later is rate-limited by default rather than unlimited.
        if channel == ActionChannel.PAYMENT:
            if context is not None:
                # Real history from the execution_attempts ledger.
                attempt_count = context.prior_payment_attempts
                in_cooldown = context.last_payment_attempt_at is not None and (
                    now - _aware(context.last_payment_attempt_at)
                ).total_seconds() <= (self.policy.payment_retry_cooldown_hours * 3600)
            else:
                # Legacy proxy: count linked failure events (kept so unit tests
                # that pass hand-built cases without a ledger still hold).
                recent = [
                    e
                    for e in case.linked_events
                    if (now - e.timestamp).total_seconds()
                    <= (self.policy.payment_retry_cooldown_hours * 3600)
                ]
                attempt_count = len(case.linked_events)
                in_cooldown = len(recent) > 0

            if attempt_count >= self.policy.payment_max_retries:
                r = RuleResult(
                    rule_name="PaymentMaxRetries",
                    passed=False,
                    details="Maximum payment retries exceeded.",
                )
                rules_evaluated.append(r)
                failed_rules.append(r)
                status = PolicyDecisionStatus.DENIED
                reason = r.details
            else:
                rules_evaluated.append(
                    RuleResult(
                        rule_name="PaymentMaxRetries", passed=True, details="Under max retries."
                    )
                )

            if in_cooldown:
                r = RuleResult(
                    rule_name="PaymentRetryCooldown",
                    passed=False,
                    details="Active cooldown period.",
                )
                rules_evaluated.append(r)
                failed_rules.append(r)
                if (
                    status != PolicyDecisionStatus.DENIED
                    and status != PolicyDecisionStatus.ESCALATE
                ):
                    status = PolicyDecisionStatus.WAIT
                    reason = r.details
            else:
                rules_evaluated.append(
                    RuleResult(
                        rule_name="PaymentRetryCooldown", passed=True, details="No active cooldown."
                    )
                )

        elif channel == ActionChannel.COMMUNICATION:
            if context is not None:
                comms = context.comms_in_last_24h
            else:
                comms = len(
                    [
                        e
                        for e in case.linked_events
                        if (now - e.timestamp).total_seconds() <= 24 * 3600
                    ]
                )
            # `comms` counts messages ALREADY sent in the window, so the cap is reached
            # at >=, not >. With `>` a max of 2 actually permitted a third message
            # (verified at runtime before this fix). PaymentMaxRetries above already
            # uses >=; this makes the two limits consistent with their stated maxima.
            if comms >= self.policy.communication_max_messages_24h:
                r = RuleResult(
                    rule_name="CommunicationMaxMessages",
                    passed=False,
                    details="Max communications exceeded in 24h.",
                )
                rules_evaluated.append(r)
                failed_rules.append(r)
                if (
                    status != PolicyDecisionStatus.DENIED
                    and status != PolicyDecisionStatus.ESCALATE
                ):
                    status = PolicyDecisionStatus.WAIT
                    reason = r.details
            else:
                rules_evaluated.append(
                    RuleResult(
                        rule_name="CommunicationMaxMessages",
                        passed=True,
                        details="Within communication limit.",
                    )
                )

        # 5. Domain-specific rules (extension point; stricter only).
        for rule in self.domain_rules.get(case.risk_category, ()):
            result = rule(case, context, self.policy)
            if result is None:
                continue
            r, status_if_failed = result
            rules_evaluated.append(r)
            if not r.passed:
                failed_rules.append(r)
                if _PRECEDENCE[status_if_failed] > _PRECEDENCE[status]:
                    status = status_if_failed
                    reason = r.details

        return PolicyDecision(
            decision_id=f"pol_{uuid.uuid4().hex[:8]}",
            status=status,
            policy_version=self.policy.version,
            rules_evaluated=rules_evaluated,
            failed_rules=failed_rules,
            reason=reason,
            contact_permission=contact_permission,
        )

    def _contact_permission(self, channel, context, status, reason, rules_evaluated, failed_rules):
        if channel == ActionChannel.PAYMENT:
            permission = context.payment_permission
            behaviour = self.policy.unknown_consent_payment
        else:
            permission = context.communication_permission
            behaviour = self.policy.unknown_consent_communication
        record = context.permission_records.get(channel.value) or {}
        evaluated = {
            "channel": channel.value,
            "status": permission,
            "record_id": record.get("record_id"),
            "unknown_consent_behaviour": behaviour if permission == "UNKNOWN" else None,
        }
        setting = f"unknown_consent_{channel.value}"

        if permission in ("OPTED_OUT", "SUPPRESSED"):
            detail = f"Customer is {permission} for {channel.value}"
            if record:
                detail += (
                    f" (record {record.get('record_id')}, source {record.get('source')}"
                    + (f": {record.get('reason')}" if record.get("reason") else "")
                    + ")"
                )
            r = RuleResult(rule_name="ContactPermission", passed=False, details=detail + ".")
            rules_evaluated.append(r)
            failed_rules.append(r)
            return PolicyDecisionStatus.DENIED, r.details, evaluated

        if permission == "ALLOWED":
            rules_evaluated.append(
                RuleResult(
                    rule_name="ContactPermission",
                    passed=True,
                    details=(
                        f"Customer permission for {channel.value} is ALLOWED "
                        f"(record {record.get('record_id')})."
                    ),
                )
            )
            return status, reason, evaluated

        unknown = f"No {channel.value} permission is recorded for this customer (UNKNOWN)"
        if behaviour == "ALLOW":
            rules_evaluated.append(
                RuleResult(
                    rule_name="ContactPermission",
                    passed=True,
                    details=(
                        f"{unknown}. Allowed by merchant policy {setting}=ALLOW - a product "
                        "setting, not recorded consent."
                    ),
                )
            )
            return status, reason, evaluated

        r = RuleResult(
            rule_name="ContactPermission",
            passed=False,
            details=f"{unknown} and merchant policy {setting}={behaviour}.",
        )
        rules_evaluated.append(r)
        failed_rules.append(r)
        outcome = (
            PolicyDecisionStatus.DENIED if behaviour == "DENY" else PolicyDecisionStatus.ESCALATE
        )
        if _PRECEDENCE[outcome] >= _PRECEDENCE[status]:
            return outcome, r.details, evaluated
        return status, reason, evaluated
