import logging
import uuid

from application.trace_ids import trace_id_for_case
from domain.models import (
    ActionType,
    ExecutionRecord,
    ExecutionStatus,
    PolicyDecisionStatus,
    RecoveryCase,
    RiskCategory,
)
from infrastructure.adapters import MockExecutionAdapter

logger = logging.getLogger(__name__)


def execution_idempotency_key(case: RecoveryCase, action) -> str:
    """The identity of one authorized dispatch: this case, this action, under this
    policy decision. Shared with api/main.py so the route and the orchestrator can
    never disagree about whether an execution has already happened."""
    return f"{case.case_id}_{getattr(action, 'value', action)}_" + (
        case.policy_decision.decision_id if case.policy_decision else "NONE"
    )


class IRecoveryAgent:
    # The recovery domain(s) this executor serves. Selection is by (domain, action):
    # an action valid in two domains (REQUEST_PAYMENT_METHOD_UPDATE) previously went to
    # whichever executor listed it first, so a subscription case was prepared by the
    # PaymentRecoveryAgent. Phase 15: the case's own domain must match.
    domains: frozenset = frozenset()

    def can_handle(self, action: ActionType) -> bool:
        raise NotImplementedError

    def prepare_execution(self, case: RecoveryCase, action: ActionType) -> dict:
        raise NotImplementedError


class PaymentRecoveryAgent(IRecoveryAgent):
    domains = frozenset({RiskCategory.FAILED_PAYMENT})

    def can_handle(self, action: ActionType) -> bool:
        return action in [
            ActionType.RETRY_PAYMENT,
            ActionType.REQUEST_PAYMENT_METHOD_UPDATE,
            ActionType.SEND_PAYMENT_REMINDER,
        ]

    def prepare_execution(self, case: RecoveryCase, action: ActionType) -> dict:
        return {
            "customer_id": case.customer_id,
            "amount": case.amount_at_risk.amount,
            "currency": case.amount_at_risk.currency,
            "reference_id": case.reference_id,
            "action": action.value,
            "agent_type": "PaymentRecoveryAgent",
        }


class CheckoutRecoveryAgent(IRecoveryAgent):
    domains = frozenset({RiskCategory.CHECKOUT_ABANDONMENT})

    def can_handle(self, action: ActionType) -> bool:
        return action in [ActionType.SEND_CHECKOUT_REMINDER, ActionType.OFFER_CHECKOUT_ASSISTANCE]

    def prepare_execution(self, case: RecoveryCase, action: ActionType) -> dict:
        return {
            "customer_id": case.customer_id,
            "action": action.value,
            "agent_type": "CheckoutRecoveryAgent",
        }


class SubscriptionRecoveryAgent(IRecoveryAgent):
    domains = frozenset({RiskCategory.FAILED_SUBSCRIPTION})

    def can_handle(self, action: ActionType) -> bool:
        return action in [
            ActionType.RETRY_BILLING,
            ActionType.SEND_SUBSCRIPTION_REMINDER,
            ActionType.REQUEST_PAYMENT_METHOD_UPDATE,
        ]

    def prepare_execution(self, case: RecoveryCase, action: ActionType) -> dict:
        return {
            "customer_id": case.customer_id,
            "action": action.value,
            "agent_type": "SubscriptionRecoveryAgent",
        }


class ReceivablesRecoveryAgent(IRecoveryAgent):
    domains = frozenset({RiskCategory.OVERDUE_INVOICE})

    def can_handle(self, action: ActionType) -> bool:
        return action in [
            ActionType.SEND_INVOICE_REMINDER,
            ActionType.SEND_PAYMENT_LINK,
            ActionType.ESCALATE_COLLECTION,
        ]

    def prepare_execution(self, case: RecoveryCase, action: ActionType) -> dict:
        return {
            "customer_id": case.customer_id,
            "amount": case.amount_at_risk.amount,
            "currency": case.amount_at_risk.currency,
            "reference_id": case.reference_id,
            "action": action.value,
            "agent_type": "ReceivablesRecoveryAgent",
        }


class PromiseRecoveryAgent(IRecoveryAgent):
    domains = frozenset({RiskCategory.BROKEN_PROMISE})

    def can_handle(self, action: ActionType) -> bool:
        return action in [
            ActionType.SEND_PROMISE_REMINDER,
            ActionType.REQUEST_NEW_COMMITMENT,
            ActionType.ESCALATE_COLLECTION,
        ]

    def prepare_execution(self, case: RecoveryCase, action: ActionType) -> dict:
        return {
            "customer_id": case.customer_id,
            "reference_id": case.reference_id,
            "action": action.value,
            "agent_type": "PromiseRecoveryAgent",
        }


class AgentOrchestrator:
    def __init__(self, execution_adapter=MockExecutionAdapter()):
        self.adapter = execution_adapter
        self.agents = [
            PaymentRecoveryAgent(),
            CheckoutRecoveryAgent(),
            SubscriptionRecoveryAgent(),
            ReceivablesRecoveryAgent(),
            PromiseRecoveryAgent(),
        ]

    def _is_policy_fresh(self, case: RecoveryCase) -> bool:
        # Simplistic freshness: No new events since policy was evaluated
        if not case.policy_decision:
            return False

        last_event_time = (
            max([e.timestamp for e in case.linked_events]) if case.linked_events else None
        )
        if last_event_time and last_event_time > case.policy_decision.timestamp:
            return False

        # In reality, also check if case amount changed, etc.
        return True

    def execute(
        self, case: RecoveryCase, requested_action: ActionType, repo=None
    ) -> ExecutionRecord | None:
        """Authorized execution boundary. Stamps the returned record with the journey
        trace_id / case_id and its own operation_id (== execution_id) - identifiers only;
        replayed (idempotent) records keep their original execution_id."""
        record = self._execute(case, requested_action, repo=repo)
        if record is not None:
            record = record.model_copy(
                update={
                    "trace_id": trace_id_for_case(case.case_id),
                    "case_id": case.case_id,
                    "operation_id": record.execution_id,
                }
            )
        return record

    def _execute(
        self, case: RecoveryCase, requested_action: ActionType, repo=None
    ) -> ExecutionRecord | None:
        # 0. Idempotency — replaying the same authorized action is a no-op.
        if repo is not None:
            _idem = execution_idempotency_key(case, requested_action)
            try:
                cached = repo.get_execution_record_by_key(_idem)
            except Exception:
                cached = None
            if cached:
                return ExecutionRecord(**cached)

        # 1. Verify Policy Exists and is PERMITTED
        if (
            not case.policy_decision
            or case.policy_decision.status != PolicyDecisionStatus.PERMITTED
        ):
            return ExecutionRecord(
                execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                action_type=requested_action,
                agent_type="Orchestrator",
                policy_decision_id=case.policy_decision.decision_id
                if case.policy_decision
                else "NONE",
                policy_version=case.policy_decision.policy_version
                if case.policy_decision
                else "NONE",
                parameters={},
                idempotency_key="REJECTED_POLICY",
                status=ExecutionStatus.REJECTED,
                adapter_used="NONE",
                result_metadata={"error": "Policy not PERMITTED."},
            )

        # 2. Verify Freshness
        if not self._is_policy_fresh(case):
            return ExecutionRecord(
                execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                action_type=requested_action,
                agent_type="Orchestrator",
                policy_decision_id=case.policy_decision.decision_id,
                policy_version=case.policy_decision.policy_version,
                parameters={},
                idempotency_key="REJECTED_STALE",
                status=ExecutionStatus.REJECTED,
                adapter_used="NONE",
                result_metadata={"error": "Policy decision is stale."},
            )

        # 3. Verify Exact Action Match
        # The recommended action must match the requested action, and policy must have evaluated it.
        # Since policy evaluates the top candidate, we check the top candidate.
        if case.recommendation.top_candidate.action_type != requested_action:
            return ExecutionRecord(
                execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                action_type=requested_action,
                agent_type="Orchestrator",
                policy_decision_id=case.policy_decision.decision_id,
                policy_version=case.policy_decision.policy_version,
                parameters={},
                idempotency_key="REJECTED_MISMATCH",
                status=ExecutionStatus.REJECTED,
                adapter_used="NONE",
                result_metadata={
                    "error": "Requested action does not match policy-authorized action."
                },
            )

        # 4. Select Agent
        selected_agent = None
        for agent in self.agents:
            if agent.can_handle(requested_action) and case.risk_category in agent.domains:
                selected_agent = agent
                break

        if not selected_agent:
            return ExecutionRecord(
                execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                action_type=requested_action,
                agent_type="Orchestrator",
                policy_decision_id=case.policy_decision.decision_id,
                policy_version=case.policy_decision.policy_version,
                parameters={},
                idempotency_key="REJECTED_NO_AGENT",
                status=ExecutionStatus.REJECTED,
                adapter_used="NONE",
                result_metadata={"error": "No specialized agent found for action."},
            )

        # 5. Prepare Execution
        try:
            params = selected_agent.prepare_execution(case, requested_action)
        except Exception as e:
            return ExecutionRecord(
                execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                action_type=requested_action,
                agent_type=selected_agent.__class__.__name__,
                policy_decision_id=case.policy_decision.decision_id,
                policy_version=case.policy_decision.policy_version,
                parameters={},
                idempotency_key="FAILED_PREP",
                status=ExecutionStatus.FAILED,
                adapter_used="NONE",
                result_metadata={"error": f"Agent prep failed: {str(e)}"},
            )

        # Idempotency Key
        idem_key = execution_idempotency_key(case, requested_action)

        # 6. Claim the execution slot BEFORE the side effect.
        #
        # The adapter call below is the point of no return: with a real payment or
        # comms adapter it moves money or contacts a customer. Previously the
        # idempotency key was written AFTER the adapter ran, so two concurrent
        # requests could both miss the cache, both execute for real, and the losing
        # INSERT was discarded by a bare `except Exception: pass`. Claiming first
        # makes the primary-key conflict the arbiter, so the second request never
        # reaches the adapter at all.
        if repo is not None and hasattr(repo, "claim_execution_slot"):
            if not repo.claim_execution_slot(idem_key):
                cached = None
                try:
                    cached = repo.get_execution_record_by_key(idem_key)
                except Exception:
                    cached = None
                if cached and not getattr(repo, "is_claim_placeholder", lambda _: False)(cached):
                    return ExecutionRecord(**cached)
                # The winner is still mid-flight; report the duplicate honestly rather
                # than executing a second time or fabricating a success.
                return ExecutionRecord(
                    execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                    action_type=requested_action,
                    agent_type="Orchestrator",
                    policy_decision_id=case.policy_decision.decision_id,
                    policy_version=case.policy_decision.policy_version,
                    parameters={},
                    idempotency_key=idem_key,
                    status=ExecutionStatus.REJECTED,
                    adapter_used="NONE",
                    result_metadata={
                        "error": "Duplicate execution suppressed: this action is already in flight."
                    },
                )

        # 7. Execute (Simulated)
        try:
            adapter_result = self.adapter.execute(params)
            status = (
                ExecutionStatus.COMPLETED_SIMULATED
                if adapter_result["adapter_status"] == "COMPLETED_SIMULATED"
                else ExecutionStatus.FAILED
            )
        except Exception as e:
            adapter_result = {"error": str(e)}
            status = ExecutionStatus.FAILED

        record = ExecutionRecord(
            execution_id=f"exec_{uuid.uuid4().hex[:8]}",
            action_type=requested_action,
            agent_type=selected_agent.__class__.__name__,
            policy_decision_id=case.policy_decision.decision_id,
            policy_version=case.policy_decision.policy_version,
            parameters=params,
            idempotency_key=idem_key,
            status=status,
            adapter_used=self.adapter.__class__.__name__,
            result_metadata=adapter_result,
        )

        # 8. Record the outcome against the claim + the append-only attempt ledger.
        #
        # A failure here is NOT swallowed silently: the ledger is what the Policy
        # Engine counts retries and customer contacts against, so losing a write
        # would under-count attempts and quietly weaken the caps.
        if repo is not None:
            try:
                if hasattr(repo, "complete_execution_slot"):
                    repo.complete_execution_slot(idem_key, record.model_dump(mode="json"))
                else:
                    repo.save_idempotency_key(idem_key, record.model_dump(mode="json"))
                repo.record_execution_attempt(
                    case_id=case.case_id,
                    action_type=requested_action.value,
                    status=status.value,
                    idempotency_key=idem_key,
                    execution_id=record.execution_id,
                    agent_type=record.agent_type,
                    policy_decision_id=case.policy_decision.decision_id,
                    contact_permission=(case.policy_decision.contact_permission or {}).get(
                        "status"
                    ),
                    trace_id=trace_id_for_case(case.case_id),
                )
            except Exception as e:  # pragma: no cover - defensive
                logger.error(
                    "execution ledger write failed for case=%s action=%s key=%s: %s",
                    case.case_id,
                    requested_action.value,
                    idem_key,
                    e,
                )
                raise

        return record
