from domain.models import CaseState, PolicyDecisionStatus, RecoveryCase

# States from which the core pipeline stages (risk / diagnosis / prediction /
# recommendation / policy / execution) must never run again — the case is
# closed. Verification and STOPPED are deliberately handled separately below:
# verification is the step that *produces* these states (see
# `record_verification`), and stopping recovery is a customer opt-out / manual
# hold that must remain possible from any state, matching existing behavior.
_CLOSED_STATES = frozenset(
    {
        CaseState.FULLY_RECOVERED,
        CaseState.PARTIALLY_RECOVERED,
        CaseState.CLOSED_NOT_RECOVERED,
        CaseState.DENIED,
        CaseState.STOPPED,
        CaseState.CANCELLED,
    }
)

# Once a case reaches one of these, an outcome event can still be RECORDED against it
# (money is real, or a further cancellation report is inert) but the case's own state
# must never move again: every cent of the obligation is accounted for. Every other
# closed state can still receive a late outcome event that moves it here (a payment
# that arrives after STOPPED, DENIED or CLOSED_NOT_RECOVERED, or a top-up that finishes
# a PARTIALLY_RECOVERED balance) - see record_outcome_position below.
OUTCOME_FINAL_STATES = frozenset({CaseState.FULLY_RECOVERED, CaseState.CANCELLED})

# States meaning "a recovery action has already gone out for this case".
#
# These are NOT closed - verification still has to run, which is why they are kept
# out of _CLOSED_STATES. But re-running the pre-execution stages from here rewinds a
# case that has already acted: a second failure event for the same obligation
# arriving on the batch-ingest path used to reset PENDING_VERIFICATION back to
# POLICY_EVALUATED (reproduced at runtime), after which another execute could
# overwrite the successful ExecutionRecord with a rejected one and verification would
# then attribute the financial outcome to an execution that never ran. An action that
# has been dispatched must be verified - or the case stopped - before the case may be
# assessed again.
_ACTION_DISPATCHED_STATES = frozenset(
    {
        CaseState.EXECUTION_PREPARING,
        CaseState.EXECUTING,
        CaseState.PENDING_VERIFICATION,
        CaseState.VERIFYING,
    }
)

# The single definition of "this case may not be reprocessed". Previously three
# modules each carried their own idea of it (domain.lifecycle, case_engine's
# ACTIVE_STATES, and case_pipeline.advance's inline three-state tuple), and the
# narrowest one produced an unhandled HTTP 500.
NON_REPROCESSABLE_STATES = _CLOSED_STATES | _ACTION_DISPATCHED_STATES


class CaseClosedError(ValueError):
    """Raised when a route tries to re-run a pipeline stage on a case that has
    already reached a closed (terminal) state, or that has already dispatched a
    recovery action and is awaiting verification."""


class CaseLifecycleManager:
    @staticmethod
    def initialize_case(case: RecoveryCase) -> None:
        if case.current_state != CaseState.DETECTED:
            raise ValueError("Case must start in DETECTED state.")
        case.current_state = CaseState.OPEN

    @staticmethod
    def move_to_diagnosing(case: RecoveryCase) -> None:
        valid_from = [CaseState.OPEN, CaseState.WAITING]
        if case.current_state not in valid_from:
            raise ValueError(f"Cannot transition to DIAGNOSING from {case.current_state}")
        case.current_state = CaseState.DIAGNOSING

    @staticmethod
    def record_diagnosis(case: RecoveryCase) -> None:
        if case.current_state != CaseState.DIAGNOSING:
            raise ValueError(f"Cannot transition to ASSESSED from {case.current_state}")
        case.current_state = CaseState.ASSESSED

    @staticmethod
    def record_prediction_and_action(case: RecoveryCase) -> None:
        if case.current_state != CaseState.ASSESSED:
            raise ValueError(f"Cannot transition to ACTION_PROPOSED from {case.current_state}")
        case.current_state = CaseState.ACTION_PROPOSED

    @staticmethod
    def submit_for_policy_review(case: RecoveryCase) -> None:
        if case.current_state != CaseState.ACTION_PROPOSED:
            raise ValueError(f"Cannot transition to POLICY_REVIEW from {case.current_state}")
        case.current_state = CaseState.POLICY_REVIEW

    @staticmethod
    def apply_policy_decision(case: RecoveryCase) -> None:
        if case.current_state != CaseState.POLICY_REVIEW:
            raise ValueError("Cannot apply policy unless in POLICY_REVIEW")

        if not case.policy_decision:
            raise ValueError("No policy decision found on case")

        status = case.policy_decision.status
        if status == PolicyDecisionStatus.PERMITTED:
            case.current_state = CaseState.APPROVED
        elif status == PolicyDecisionStatus.DENIED:
            case.current_state = CaseState.DENIED
        elif status == PolicyDecisionStatus.WAIT:
            case.current_state = CaseState.WAITING
        elif status == PolicyDecisionStatus.ESCALATE:
            case.current_state = CaseState.ESCALATED

    # ------------------------------------------------------------------
    # Live-pipeline transitions.
    #
    # The five methods above model the OPEN -> DIAGNOSING -> ASSESSED ->
    # ACTION_PROPOSED -> POLICY_REVIEW family and are only actually driven by
    # `application/case_engine.py` (case creation) today. The routes in
    # `api/main.py` and `application/case_pipeline.py` run a *different*,
    # already-shipped stage order (assess-risk -> diagnose -> predict ->
    # recommend -> policy-check -> execute -> verify) and, before this change,
    # mutated `case.current_state` directly with no guard at all. These
    # methods are the guarded equivalent for that live order — same target
    # states the routes already used, now with one real protection: a closed
    # (terminal) case can no longer be silently reprocessed and overwritten.
    # No new CaseState values, no change to any transition that already runs
    # in production.
    # ------------------------------------------------------------------

    @staticmethod
    def assert_reprocessable(case: RecoveryCase, what: str) -> None:
        """Guard for any stage that must not run on a case which is closed, or whose
        recovery action has already been dispatched and is awaiting verification.

        Exposed publicly so a read-mostly stage (prediction) can enforce the same rule
        without pretending to be a state transition."""
        if case.current_state in _CLOSED_STATES:
            raise CaseClosedError(
                f"Cannot {what}: case {case.case_id} is already {case.current_state.value}."
            )
        if case.current_state in _ACTION_DISPATCHED_STATES:
            raise CaseClosedError(
                f"Cannot {what}: case {case.case_id} has already dispatched a recovery "
                f"action and is {case.current_state.value}; verify the outcome (or stop "
                f"the case) before reassessing it."
            )

    @staticmethod
    def move_to_stopped(case: RecoveryCase) -> None:
        """Halt automated recovery.

        Permitted from any state a case can still be worked in - a customer opt-out,
        dispute or manual hold can arrive at any point, including after an action has
        been dispatched. Refused only once the case is CLOSED: stamping STOPPED over a
        settled case would discard the terminal state verification produced, leaving a
        case whose state says "halted" while its outcome says the money came back.
        Stopping an already-STOPPED case is a no-op, so retries stay safe.
        """
        if case.current_state == CaseState.STOPPED:
            return
        if case.current_state in _CLOSED_STATES:
            raise CaseClosedError(
                f"Cannot stop case {case.case_id}: it is already "
                f"{case.current_state.value} and has a settled outcome."
            )
        case.current_state = CaseState.STOPPED

    @staticmethod
    def record_risk_assessment(case: RecoveryCase) -> None:
        CaseLifecycleManager.assert_reprocessable(case, "assess risk")
        case.current_state = CaseState.ASSESSED

    @staticmethod
    def record_diagnosis_result(case: RecoveryCase) -> None:
        CaseLifecycleManager.assert_reprocessable(case, "diagnose")
        case.current_state = CaseState.DIAGNOSING

    @staticmethod
    def record_recommendation_result(case: RecoveryCase) -> None:
        CaseLifecycleManager.assert_reprocessable(case, "recommend an action")
        case.current_state = CaseState.RECOMMENDING

    @staticmethod
    def record_policy_evaluation(case: RecoveryCase) -> None:
        CaseLifecycleManager.assert_reprocessable(case, "run the policy check")
        case.current_state = CaseState.POLICY_EVALUATED

    @staticmethod
    def record_execution_result(case: RecoveryCase, to_state: CaseState) -> None:
        if case.current_state in _CLOSED_STATES:
            raise CaseClosedError(
                f"Cannot execute an action: case {case.case_id} is already "
                f"{case.current_state.value}."
            )
        case.current_state = to_state

    @staticmethod
    def record_verification_result(case: RecoveryCase, to_state: CaseState) -> None:
        """Apply the state produced by a completed verification.

        Deliberately NOT guarded by `_CLOSED_STATES`: today's design allows a
        case to be verified even off a non-`COMPLETED_SIMULATED` execution
        (see `tests/test_api_endpoints.py`), and *repeat* verification is
        prevented upstream by outcome idempotency (`api/main.py:verify_outcome`
        returns the persisted outcome before this is ever called again) rather
        than by case state. Changing that here would alter existing, tested
        business behavior, which this phase must not do.
        """
        case.current_state = to_state

    @staticmethod
    def record_outcome_position(case: RecoveryCase, to_state: CaseState) -> None:
        """Apply the case state that follows from reconciling outcome events
        (application/outcome_events.py) against the case's financial position.

        Unlike record_verification_result, this IS guarded, by exactly one rule: a case
        already at FULLY_RECOVERED or CANCELLED never moves again - every cent of the
        obligation is accounted for, so a further outcome event can still be recorded
        (as excess or a no-op) but cannot change the case's state. Every other state,
        including every other closed state, MAY move here: an outcome event is
        independent of the action pipeline's reprocessing guards, because money can
        arrive (or an obligation can be cancelled) whether or not - or before - any
        recovery action was ever attempted, and a late payment on a STOPPED, DENIED or
        CLOSED_NOT_RECOVERED case, or a top-up that finishes a PARTIALLY_RECOVERED
        balance, is real recovered money that must be reflected (Phase 5 "full
        recovery" and "late recovery" requirements).
        """
        if case.current_state in OUTCOME_FINAL_STATES:
            if to_state != case.current_state:
                raise CaseClosedError(
                    f"Cannot move case {case.case_id} to {to_state.value}: it is already "
                    f"{case.current_state.value} and every outcome event for this "
                    "obligation is already accounted for."
                )
            return
        case.current_state = to_state

    @staticmethod
    def record_human_review(case: RecoveryCase, approved: bool) -> None:
        """Existing behavior: approval clears the case for execution;
        rejection leaves `current_state` exactly as it was (unchanged)."""
        if approved:
            case.current_state = CaseState.POLICY_REVIEW
