"""
CasePipelineService — runs the deterministic case lifecycle end to end.

Stages: assess-risk -> diagnose -> predict-recovery -> recommend-action ->
policy-check. Execution and verification stay separate (they have side effects
and their own guards). Each stage is idempotent-ish: calling it again recomputes
and overwrites that stage's result.

The individual API routes in api/main.py still exist for step-by-step control;
this service is what `POST /events/batch?auto=true` and
`POST /cases/{id}/advance` use so a submitted event actually moves.
"""

import uuid
from datetime import UTC, datetime

from domain.lifecycle import NON_REPROCESSABLE_STATES, CaseLifecycleManager
from domain.models import (
    CaseState,
    Money,
    RecoveryCase,
    RecoveryPrediction,
)


def predict_recovery_for_case(
    case: RecoveryCase, dataset_id: str | None = None, predictor=None
) -> RecoveryPrediction:
    """
    Shadow ML prediction with a deterministic fallback.

    If a dataset-isolated model exists it is used (shadow-only). Otherwise we fall
    back to DeterministicBaselinePredictor so downstream ERV / ranking is never
    driven by a meaningless 0.0.

    `predictor` lets a caller that processes many rows for the same dataset (e.g.
    /datasets/{id}/generate-cases) build one MLPaymentFailurePredictor and reuse
    it, instead of re-scanning the model registry and re-loading the model on
    every row. When omitted the behaviour is unchanged.

    `dataset_id`, when omitted, is auto-resolved from the case's own linked
    event (`RevenueEvent.source_ref` — "dataset_id for file replay" per
    infrastructure/orm.py's own column comment). Found and fixed during the
    AI reconstruction: a caller of this function (or of POST
    /cases/{id}/predict-recovery, which the frontend never calls with
    ?dataset_id=... — it isn't called by the frontend at all) that omitted
    dataset_id always fell back to the deterministic baseline even when a
    trained, calibrated model already existed for that exact case's
    dataset. See docs/ai_reconstruction/LIVE_MODEL_FORENSIC_AUDIT.md §2.
    An explicitly passed dataset_id still wins, so existing callers that
    already know it (e.g. shared_predictor at case-generation time) are
    unaffected.
    """
    from application.recovery_predictor import DeterministicBaselinePredictor
    from application.recovery_predictor_ml import MLPaymentFailurePredictor

    # Start from the raw source row (dataset replay) so a per-dataset model sees
    # the same original columns it was trained on; canonical keys are layered on
    # top as a fallback for models that only know canonical inputs.
    src_event = case.linked_events[-1] if getattr(case, "linked_events", None) else None
    if dataset_id is None and predictor is None and src_event is not None:
        dataset_id = src_event.source_ref
    row = dict(src_event.raw_payload) if src_event and src_event.raw_payload else {}
    ts = (
        src_event.timestamp.isoformat()
        if src_event and src_event.timestamp
        else datetime.now(UTC).isoformat()
    )

    feature_dict = {
        **row,
        "AMOUNT": case.amount_at_risk.amount if case.amount_at_risk else 0.0,
        "BALANCE": case.amount_at_risk.amount if case.amount_at_risk else 0.0,
        "CUSTOMER_ID": case.customer_id,
        "ACCOUNT_ID": case.customer_id,
        "ENTITY_ID": case.customer_id,
        "TIMESTAMP": ts,
    }

    prob = None
    status = "SUCCESS"
    model_version = "unknown"
    explanation = None
    confidence_tier = None
    anomaly = None

    if predictor is None:
        try:
            predictor = MLPaymentFailurePredictor(dataset_id=dataset_id)
        except Exception:
            # Phase 11 fix: construction itself loads the joblib/XGBoost
            # artifact from disk (MLPaymentFailurePredictor.__init__) and was
            # NOT covered by the try/except below, which only wraps the
            # predict_failure_risk() call. A corrupted or unreadable registry
            # artifact therefore crashed the whole case pipeline instead of
            # falling back to the deterministic baseline like every other
            # shadow-prediction failure mode. Same safety contract as the
            # predict-time except below: any ML failure here is advisory-only.
            baseline = DeterministicBaselinePredictor().predict(case)
            return baseline
    if predictor.model is not None:
        try:
            res = predictor.predict_failure_risk(feature_dict)
            if res.get("status") == "SUCCESS":
                # model predicts *failure* risk; recovery probability is the complement
                prob = 1.0 - float(res.get("probability", 0.0))
                status = "SHADOW_ONLY"
                meta = res.get("model_metadata", {})
                model_version = meta.get("model_version", "shadow")
                explanation = meta.get("explanation")
                confidence_tier = meta.get("confidence_tier")
                anomaly = meta.get("anomaly")
            # INVALID_PROBABILITY (Phase 12) falls through with prob still
            # None, exactly like NO_MODEL/incomplete-features - same safe
            # fallback to the deterministic baseline below.
        except ValueError:
            prob = None  # incomplete features -> fall through to baseline
        except Exception:
            # Phase 7 safety contract: an AI/ML failure of ANY kind (a corrupted
            # model file, a native-library crash, an unexpected data shape) must
            # never take the case pipeline down with it. Shadow prediction is
            # advisory only - any error here falls through to the deterministic
            # baseline below, exactly like "no model available" does.
            prob = None

    if prob is None:
        baseline = DeterministicBaselinePredictor().predict(case)
        return baseline

    return RecoveryPrediction(
        prediction_id=f"pred_{uuid.uuid4().hex[:8]}",
        recovery_probability=max(0.0, min(1.0, prob)),
        confidence=0.8,
        model_version=model_version,
        feature_version="1.0",
        prediction_timestamp=datetime.now(UTC),
        contributing_features=feature_dict,
        prediction_status=status,
        explanation=explanation,
        confidence_tier=confidence_tier,
        anomaly=anomaly,
    )


class CasePipelineService:
    def __init__(self, repo, audit):
        self.repo = repo
        self.audit = audit

    # -- individual stages ------------------------------------------------
    def assess_risk(self, case: RecoveryCase) -> RecoveryCase:
        from application.risk_detector import DeterministicRiskDetector

        assessment = DeterministicRiskDetector().assess_risk(case)
        # Record the state the case was ACTUALLY in, not a hard-coded literal. The
        # literals happened to be right on a first pass and were simply false on any
        # re-run - and this is the audit trail for /datasets/{id}/generate-cases,
        # /events/batch and /advance, i.e. essentially every case in the product.
        prior = case.current_state.value
        CaseLifecycleManager.record_risk_assessment(case)
        case.risk_assessment = assessment
        self.audit.log_transition(
            case_id=case.case_id,
            from_state=prior,
            to_state=CaseState.ASSESSED.value,
            evidence={
                "action": "risk_assessment",
                "score": assessment.score,
                "level": assessment.risk_level,
            },
        )
        return case

    def diagnose(self, case: RecoveryCase) -> RecoveryCase:
        from application.diagnosis_engine import DeterministicDiagnosisEngine

        diagnosis = DeterministicDiagnosisEngine().diagnose(case)
        prior = case.current_state.value
        CaseLifecycleManager.record_diagnosis_result(case)
        case.diagnosis = diagnosis
        self.audit.log_transition(
            case_id=case.case_id,
            from_state=prior,
            to_state=CaseState.DIAGNOSING.value,
            evidence={
                "action": "diagnosis",
                "cause": diagnosis.cause_category,
                "confidence": diagnosis.confidence,
            },
        )
        return case

    def predict(
        self, case: RecoveryCase, dataset_id: str | None = None, predictor=None
    ) -> RecoveryCase:
        prediction = predict_recovery_for_case(case, dataset_id=dataset_id, predictor=predictor)
        case.prediction = prediction
        # Phase 12 prediction audit trail: feature NAMES/contributions/direction
        # are recorded for "why did this prediction happen" auditability; raw
        # feature VALUES are deliberately dropped here (kept only on the
        # in-memory RecoveryPrediction.explanation, not persisted to the audit
        # log) - the audit trail needs "amount mattered more than reminder
        # count", not a permanent record of the customer's exact amount.
        explanation_summary = (
            [{"feature": c["feature"], "direction": c["direction"]} for c in prediction.explanation]
            if prediction.explanation
            else None
        )
        self.audit.log_transition(
            case_id=case.case_id,
            from_state=case.current_state.value,
            to_state=case.current_state.value,
            evidence={
                "action": "recovery_prediction",
                "probability": prediction.recovery_probability,
                "model_version": prediction.model_version,
                "status": prediction.prediction_status,
                "confidence_tier": prediction.confidence_tier,
                "explanation_summary": explanation_summary,
                "anomaly_level": (prediction.anomaly or {}).get("level"),
                "dataset_id": dataset_id,
            },
        )
        return case

    def recommend(self, case: RecoveryCase) -> RecoveryCase:
        from application.action_evaluator import DeterministicActionEvaluator

        recommendation = DeterministicActionEvaluator().evaluate(case)
        prior = case.current_state.value
        CaseLifecycleManager.record_recommendation_result(case)
        case.recommendation = recommendation
        if recommendation.top_candidate:
            currency = case.amount_at_risk.currency if case.amount_at_risk else "INR"
            case.expected_recoverable_value = Money(
                amount=recommendation.top_candidate.expected_recoverable_value, currency=currency
            )
        top_type = (
            recommendation.top_candidate.action_type if recommendation.top_candidate else "NONE"
        )
        self.audit.log_transition(
            case_id=case.case_id,
            from_state=prior,
            to_state=CaseState.RECOMMENDING.value,
            evidence={
                "action": "action_recommendation",
                "top_candidate": str(top_type),
                "status": recommendation.status.value,
            },
        )
        self._record_specialist_agent(case, prior_state=CaseState.RECOMMENDING.value)
        return case

    def _record_specialist_agent(self, case: RecoveryCase, prior_state: str) -> None:
        """Phase 15: route the case to its domain specialist, validate the structured
        recommendation, cross-check it against the deterministic evaluator, and write
        one traceable audit record. Supplementary and fail-safe: nothing here can change
        case.recommendation, case.policy_decision or any amount, and any failure is
        swallowed (the deterministic path has already produced the recommendation)."""
        try:
            from application.decision_context import build_decision_context
            from application.specialist_agents import agent_evidence, run_specialist

            signals = None
            policy_context = None
            if hasattr(self.repo, "get_policy_context"):
                try:
                    policy_context = self.repo.get_policy_context(case.case_id)
                except Exception:
                    policy_context = None
            ctx = build_decision_context(case, policy_context)
            signals = ctx.get("policy_signals")
            result = run_specialist(case, ctx, policy_signals=signals)
            self.audit.log_transition(
                case_id=case.case_id,
                from_state=prior_state,
                to_state=prior_state,
                evidence=agent_evidence(result),
            )
        except Exception:  # pragma: no cover - the agent layer must never break the pipeline
            pass

    def policy_check(self, case: RecoveryCase) -> RecoveryCase:
        from application.policy_engine import DeterministicPolicyEngine

        context = None
        if hasattr(self.repo, "get_policy_context"):
            try:
                context = self.repo.get_policy_context(case.case_id)
            except Exception:
                context = None
        decision = DeterministicPolicyEngine().evaluate(case, context=context)
        prior = case.current_state.value
        CaseLifecycleManager.record_policy_evaluation(case)
        case.policy_decision = decision
        self.audit.log_transition(
            case_id=case.case_id,
            from_state=prior,
            to_state=CaseState.POLICY_EVALUATED.value,
            evidence={
                "action": "policy_evaluation",
                "decision": decision.status.value,
                "reason": decision.reason,
                "decision_id": decision.decision_id,
                "contact_permission": decision.contact_permission,
            },
        )
        return case

    # -- full run ------------------------------------------------------------
    def advance(
        self, case: RecoveryCase, dataset_id: str | None = None, predictor=None
    ) -> RecoveryCase:
        # One definition of "not reprocessable", shared with the lifecycle manager.
        # This inline set previously listed only three of the five closed states, so a
        # PARTIALLY_RECOVERED or DENIED case fell through to the stage guard and
        # surfaced as an unhandled HTTP 500 while a FULLY_RECOVERED case got a polite
        # no-op. It now also covers cases whose action is already dispatched, so
        # auto-advance on a second event cannot rewind a case that has acted.
        if case.current_state in NON_REPROCESSABLE_STATES:
            return case
        self.assess_risk(case)
        self.diagnose(case)
        self.predict(case, dataset_id=dataset_id, predictor=predictor)
        self.recommend(case)
        self.policy_check(case)
        self.repo.save(case)
        return case
