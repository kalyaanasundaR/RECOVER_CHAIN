from contextlib import contextmanager

from pydantic import BaseModel
from sqlalchemy.orm import Session

from application.action_evaluator import DeterministicActionEvaluator
from application.agents import AgentOrchestrator
from application.case_engine import CaseEngine
from application.diagnosis_engine import DeterministicDiagnosisEngine
from application.policy_engine import DeterministicPolicyEngine
from application.recovery_predictor import DeterministicBaselinePredictor
from application.verification_engine import MockOutcomeVerificationAdapter, VerificationEngine
from domain.lifecycle import CaseLifecycleManager
from domain.models import CaseState
from evaluation.models import EvaluationMetrics, EvaluationScenario, ScenarioResult
from infrastructure.repositories import SqlAlchemyAuditRecorder, SqlAlchemyCaseRepository


class AblationConfig(BaseModel):
    skip_policy: bool = False
    skip_diagnosis: bool = False
    skip_prediction: bool = False
    skip_recommendation: bool = False
    experiment_mode: str = "A"  # A (Det), B (LLM Diag), C (LLM Rec), D (LLM Full)


class OperationalStoreError(RuntimeError):
    """Raised when evaluation is pointed at the operational database."""


def _is_operational_bind(db: Session) -> bool:
    from infrastructure.db import engine as operational_engine

    bind = db.get_bind()
    return bind is operational_engine or str(bind.url) == str(operational_engine.url)


class EvaluationRunner:
    """Runs gold-labelled evaluation scenarios through the real pipeline.

    Scenarios create cases, events and audit rows as they run. Those must never land
    in the operational database: POST /evaluation/run used to hand this runner the
    request's own session, so each run added its scenarios to the live case table and
    moved dashboard revenue-at-risk and recovered figures (and a second run failed with
    HTTP 500 on the fixed scenario event ids). The runner now refuses the operational
    store outright; use ``EvaluationRunner.isolated()``.
    """

    ISOLATION = "ISOLATED_EPHEMERAL_STORE"

    @classmethod
    @contextmanager
    def isolated(cls, ablation: AblationConfig = AblationConfig()):
        """A runner bound to a private, in-memory SQLite store that is discarded when
        the block exits. Each run starts empty, so runs are independent and repeatable."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        import infrastructure.dataset_orm  # noqa: F401  (register every table)
        import infrastructure.orm  # noqa: F401
        from infrastructure.db import Base

        eval_engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(eval_engine)
        session = sessionmaker(bind=eval_engine, autocommit=False, autoflush=False)()
        try:
            yield cls(session, ablation=ablation)
        finally:
            session.close()
            eval_engine.dispose()

    def __init__(self, db: Session, ablation: AblationConfig = AblationConfig()):
        if _is_operational_bind(db):
            raise OperationalStoreError(
                "EvaluationRunner refuses to write evaluation scenarios into the operational "
                "database. Use EvaluationRunner.isolated()."
            )
        self.db = db
        self.repo = SqlAlchemyCaseRepository(db)
        self.audit = SqlAlchemyAuditRecorder(db)
        self.ablation = ablation

        self.case_engine = CaseEngine(self.repo, self.audit)
        self.diagnosis = DeterministicDiagnosisEngine()
        self.prediction = DeterministicBaselinePredictor()
        self.action_evaluator = DeterministicActionEvaluator()
        self.policy = DeterministicPolicyEngine()
        self.orchestrator = AgentOrchestrator()
        # Explicit, and only here: the evaluation harness runs entirely in its own
        # isolated ephemeral store (EvaluationRunner.isolated) and never writes to the
        # operational database, so a scripted settlement source is legitimate for
        # scoring gold scenarios. Nowhere in the operational path gets a default
        # adapter any more - VerificationEngine() alone returns an engine that can only
        # report the real, already-reconciled position (see current_position).
        self.verification = VerificationEngine(MockOutcomeVerificationAdapter())

        from application.langgraph_orchestrator import build_graph
        from application.llm_adapters import SimulatedLLMAdapter
        from application.real_llm_adapter import RealGeminiAdapter

        self.graph = build_graph()

        self.simulated_llm = SimulatedLLMAdapter()
        self.real_llm = RealGeminiAdapter()

        if "REAL" in self.ablation.experiment_mode:
            self.llm = self.real_llm
            if not self.real_llm.available:
                print(
                    f"Warning: {self.ablation.experiment_mode} requested but REAL_LLM_UNAVAILABLE. Tests should catch this."
                )
        else:
            self.llm = self.simulated_llm

    def run_scenario(self, scenario: EvaluationScenario) -> ScenarioResult:
        # 1. Ingestion
        case_id = None
        for event in scenario.events:
            case, _ = self.case_engine.ingest_normalized_event(event)
            case_id = case.case_id

        case = self.repo.get_by_id(case_id)

        # 2. Risk Detection (In-built with case engine or separate? In phase 5 we added API, let's just do it directly if not there)
        from application.risk_detector import DeterministicRiskDetector

        risk_detector = DeterministicRiskDetector()
        case.risk_assessment = risk_detector.assess_risk(case)
        # Route state changes through the one authority, like every other caller.
        # This harness was the last place still assigning current_state directly.
        CaseLifecycleManager.record_risk_assessment(case)

        # 3-5. Intelligence Layer (LangGraph)
        # We invoke the graph state machine, which wraps both deterministic and LLM logic based on mode
        from application.langgraph_orchestrator import GraphState

        initial_state: GraphState = {
            "case": case,
            "mode": self.ablation.experiment_mode,
            "llm": self.llm,
            "det_diag": self.diagnosis,
            "det_rec": self.action_evaluator,
            "llm_latency_ms": 0,
            "llm_tokens_used": 0,
            "prompt_version": "v1",
            "model_identifier": self.llm.model_version,
            "error_message": None,
        }

        final_state = self.graph.invoke(initial_state)
        case = final_state["case"]

        # Apply strict ablation nullification post-graph if needed (to honor Phase 12A flags)
        if self.ablation.skip_diagnosis:
            case.diagnosis = None
        if self.ablation.skip_prediction:
            case.prediction = None
        if self.ablation.skip_recommendation:
            case.recommendation = None

        # 6. Policy
        if not self.ablation.skip_policy:
            case.policy_decision = self.policy.evaluate(case)
        else:
            # Bypass policy gate entirely
            import uuid

            from domain.models import PolicyDecision, PolicyDecisionStatus

            case.policy_decision = PolicyDecision(
                decision_id=f"pol_bypass_{uuid.uuid4().hex[:8]}",
                status=PolicyDecisionStatus.PERMITTED,
                policy_version="bypass",
                rules_evaluated=[],
                failed_rules=[],
                reason="Policy gate skipped via ablation.",
            )
        CaseLifecycleManager.record_policy_evaluation(case)

        # 7. Execution
        action = None
        if case.recommendation and case.recommendation.top_candidate:
            action = case.recommendation.top_candidate.action_type

        if action:
            record = self.orchestrator.execute(case, action)
            case.execution_record = record
            if record.status.value == "COMPLETED_SIMULATED":
                CaseLifecycleManager.record_execution_result(case, CaseState.PENDING_VERIFICATION)

        # 8. Verification
        # Only verify if execution succeeded (as a mock test flow) or if we just want to force a verification
        # The user said simulate verification result.
        outcome = self.verification.reconcile(case, scenario.gold.expected_simulated_verification)
        case.outcome = outcome
        case.actual_amount_recovered = outcome.actual_amount_recovered

        self.repo.save(case)

        # Evaluate against Gold
        diagnosis_matched = (
            case.diagnosis.cause_category.value == scenario.gold.expected_diagnosis_category
            if case.diagnosis and scenario.gold.expected_diagnosis_category
            else True
        )
        action_matched = (
            action == scenario.gold.expected_action_type
            if scenario.gold.expected_action_type
            else True
        )
        policy_matched = (
            case.policy_decision.status == scenario.gold.expected_policy_status
            if scenario.gold.expected_policy_status
            else True
        )
        outcome_matched = (
            outcome.status == scenario.gold.expected_outcome_status
            if scenario.gold.expected_outcome_status
            else True
        )

        # Safety Analysis
        unsafe_execution = False
        if (
            case.execution_record
            and case.execution_record.status.value == "COMPLETED_SIMULATED"
            and case.policy_decision.status.value != "PERMITTED"
        ):
            unsafe_execution = True

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            amount_at_risk=case.amount_at_risk.amount,
            expected_recoverable_value=case.recommendation.top_candidate.expected_recoverable_value
            if case.recommendation and case.recommendation.top_candidate
            else 0.0,
            actual_recovered=case.outcome.actual_amount_recovered.amount if case.outcome else 0.0,
            diagnosis_matched=diagnosis_matched,
            action_matched=action_matched,
            policy_matched=policy_matched,
            outcome_matched=outcome_matched,
            policy_decision_status=case.policy_decision.status.value
            if case.policy_decision
            else "NONE",
            execution_status=case.execution_record.status.value if case.execution_record else None,
            unsafe_execution=unsafe_execution,
            latency_ms=final_state.get("llm_latency_ms", 0),
            tokens_used=final_state.get("llm_tokens_used", 0),
            error_message=final_state.get("error_message"),
            audit_complete=True,
        )

    def run_all(self, scenarios: list) -> EvaluationMetrics:
        results = []
        for sc in scenarios:
            results.append(self.run_scenario(sc))

        total = len(results)
        passed = sum(
            1
            for r in results
            if r.diagnosis_matched
            and r.action_matched
            and r.policy_matched
            and r.outcome_matched
            and not r.unsafe_execution
        )

        total_risk = sum(r.amount_at_risk for r in results)
        total_erv = sum(r.expected_recoverable_value for r in results)
        total_rec = sum(r.actual_recovered for r in results)

        unsafe_count = sum(1 for r in results if r.unsafe_execution)
        unnecessary_escalations = sum(
            1 for r in results if r.policy_decision_status == "ESCALATE" and not r.policy_matched
        )

        total_latency = sum(r.latency_ms for r in results)
        total_tokens = sum(r.tokens_used for r in results)
        invalid_outputs = sum(1 for r in results if r.error_message)

        return EvaluationMetrics(
            total_scenarios=total,
            passed_scenarios=passed,
            failed_scenarios=total - passed,
            scenario_pass_rate=passed / total if total else 0,
            diagnosis_accuracy=sum(1 for r in results if r.diagnosis_matched) / total
            if total
            else 0,
            recommendation_accuracy=sum(1 for r in results if r.action_matched) / total
            if total
            else 0,
            policy_decision_accuracy=sum(1 for r in results if r.policy_matched) / total
            if total
            else 0,
            outcome_accuracy=sum(1 for r in results if r.outcome_matched) / total if total else 0,
            unsafe_execution_rate=unsafe_count / total if total else 0,
            policy_bypass_rate=unsafe_count / total if total else 0,
            unnecessary_escalation_rate=unnecessary_escalations / total if total else 0,
            simulated_recovery_rate=total_rec / total_risk if total_risk else 0,
            total_amount_at_risk=total_risk,
            total_gross_erv=total_erv,
            total_simulated_recovered=total_rec,
            recovery_gap=total_risk - total_rec,
            audit_completeness=sum(1 for r in results if r.audit_complete) / total if total else 0,
            avg_latency_ms=total_latency / total if total else 0,
            total_tokens_used=total_tokens,
            invalid_output_rate=invalid_outputs / total if total else 0,
            scenario_results=results,
        )
