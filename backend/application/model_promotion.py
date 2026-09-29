"""Phase 14 — champion/challenger comparison, the promotion gate, production-
validation safety, retraining readiness, and the pre-retraining data-
sufficiency gate.

Nothing in this module executes, retrains, deploys, or promotes anything.
Every function returns a recommendation/status string plus the evidence
behind it; moving a model between lifecycle stages is a decision this module
can only ever recommend, never perform - the model registry's `status`/
`lifecycle_status` fields (application/ml_training.py, Phase 11) are still
written only by the training pipeline itself, and a stage beyond
QUALITY_PASSED requires an explicit `reviewed_by` string (a human identifier)
passed in by the caller - there is no code path here that manufactures one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

from application.prediction_outcome_linkage import PredictionOutcomeRecord

# Ordered lifecycle - a model can only ever be reported at the highest stage
# its ACTUAL evidence justifies; this function never skips ahead on request.
LIFECYCLE_STAGES = [
    "TRAINED",
    "EVALUATED",
    "QUALITY_PASSED",
    "CHALLENGER",
    "PROMOTION_REVIEW",
    "APPROVED",
    "SHADOW",
    "PRODUCTION_VALIDATED",
]

# Operational heuristic, documented (Phase 11 already used this exact number
# for the NBA/root-cause training floor) - reused here, not reinvented, for
# "enough production evidence to even consider PRODUCTION_VALIDATED".
MIN_PRODUCTION_EVIDENCE_ROWS = 200


@dataclass
class ChampionChallengerComparison:
    champion_model_id: str
    challenger_model_id: str
    metric_deltas: dict = field(default_factory=dict)
    challenger_wins: list = field(default_factory=list)
    champion_wins: list = field(default_factory=list)
    inconclusive: list = field(default_factory=list)
    recommendation: str = ""
    auto_promoted: bool = False  # ALWAYS False - documents the guarantee, not a toggle


def compare_champion_challenger(
    champion_meta: dict, challenger_meta: dict
) -> ChampionChallengerComparison:
    """Compares two real registry metadata dicts (application/ml_training.py's
    output) across every metric the brief lists. Never declares a winner from
    one metric alone - `recommendation` only says CHALLENGER_BETTER when it
    wins on accuracy/quality AND does not clearly lose on calibration or
    operational cost; otherwise INCONCLUSIVE, explicitly."""
    comp = ChampionChallengerComparison(
        champion_model_id=champion_meta.get("model_id", "unknown"),
        challenger_model_id=challenger_meta.get("model_id", "unknown"),
    )

    def _gate(meta: dict) -> dict:
        return meta.get("quality_gate", {}) or {}

    champion_metrics = _gate(champion_meta)
    challenger_metrics = _gate(challenger_meta)

    # Higher-is-better metrics.
    for metric in ("roc_auc",):
        c0, c1 = champion_metrics.get(metric), challenger_metrics.get(metric)
        if c0 is None or c1 is None:
            comp.inconclusive.append(metric)
            continue
        comp.metric_deltas[metric] = round(c1 - c0, 4)
        (
            comp.challenger_wins
            if c1 > c0
            else comp.champion_wins
            if c0 > c1
            else comp.inconclusive
        ).append(metric)

    champion_test = champion_meta.get("final_test_metrics", {}) or {}
    challenger_test = challenger_meta.get("final_test_metrics", {}) or {}
    for metric, higher_is_better in (
        ("pr_auc", True),
        ("brier_score", False),
        ("f1", True),
    ):
        c0, c1 = champion_test.get(metric), challenger_test.get(metric)
        if c0 is None or c1 is None:
            comp.inconclusive.append(metric)
            continue
        comp.metric_deltas[metric] = round(c1 - c0, 4)
        challenger_better = (c1 > c0) if higher_is_better else (c1 < c0)
        (
            comp.challenger_wins
            if challenger_better
            else comp.champion_wins
            if c1 != c0
            else comp.inconclusive
        ).append(metric)

    # Calibration status (a flag, not a number) and operational cost (artifact
    # size, a real proxy for load/serve cost - inference latency isn't stored
    # in registry metadata today, disclosed as UNKNOWN rather than guessed).
    comp.metric_deltas["calibrated_champion"] = champion_meta.get("calibrated")
    comp.metric_deltas["calibrated_challenger"] = challenger_meta.get("calibrated")
    comp.metric_deltas["inference_latency"] = "UNKNOWN - not captured in registry metadata"

    champion_size = _artifact_size(champion_meta)
    challenger_size = _artifact_size(challenger_meta)
    if champion_size is not None and challenger_size is not None:
        comp.metric_deltas["model_size_bytes_delta"] = challenger_size - champion_size

    # Decision rule, explicit and conservative: challenger must win on more
    # metrics than it loses, AND lose on none of the metrics that matter most
    # for THIS system's stated priority (financial safety over raw accuracy -
    # so a Brier-score loss is disqualifying even with a higher ROC-AUC).
    if "brier_score" in comp.champion_wins:
        comp.recommendation = (
            "INCONCLUSIVE - challenger's calibration (Brier score) is worse than the "
            "champion's; a higher ROC-AUC does not offset a worse-calibrated ERV input"
        )
    elif len(comp.challenger_wins) > len(comp.champion_wins) and not comp.champion_wins:
        comp.recommendation = "CHALLENGER_BETTER_ON_AVAILABLE_METRICS - recommend PROMOTION_REVIEW"
    elif len(comp.champion_wins) > len(comp.challenger_wins):
        comp.recommendation = "CHAMPION_REMAINS_BETTER"
    else:
        comp.recommendation = "INCONCLUSIVE - mixed results, human review required"

    comp.auto_promoted = False
    return comp


def _artifact_size(meta: dict) -> int | None:
    path = meta.get("artifact_path")
    if not path:
        return None
    for candidate in (path, os.path.join("backend", path)):
        if os.path.exists(candidate):
            return os.path.getsize(candidate)
    return None


def promotion_gate(
    meta: dict,
    *,
    is_challenger_evaluated: bool = False,
    promotion_review_requested: bool = False,
    reviewed_by: str | None = None,
    is_shadow_serving: bool = False,
    production_evidence_rows: int = 0,
) -> dict:
    """Returns the highest lifecycle stage the SUPPLIED evidence justifies,
    plus the exact reason it did not go further. Every stage past
    QUALITY_PASSED requires an explicit signal from the caller - this
    function has no code path that invents `reviewed_by` or flips
    `promotion_review_requested`/`is_shadow_serving` on its own."""
    quality_gate = meta.get("quality_gate", {}) or {}
    stage = "TRAINED"  # an artifact + metadata existing is always at least this
    reasons = ["artifact and metadata exist"]

    if meta.get("final_test_metrics"):
        stage = "EVALUATED"
        reasons.append("test-set metrics were computed")

    if quality_gate.get("passed"):
        stage = "QUALITY_PASSED"
        reasons.append("quality gate passed")
    else:
        return {
            "stage": stage,
            "reasons": reasons + [f"quality gate failed: {quality_gate.get('reasons')}"],
            "blocked_at": "QUALITY_PASSED",
        }

    if is_challenger_evaluated:
        stage = "CHALLENGER"
        reasons.append("evaluated head-to-head against a champion")
    else:
        return {"stage": stage, "reasons": reasons, "blocked_at": "CHALLENGER"}

    if promotion_review_requested:
        stage = "PROMOTION_REVIEW"
        reasons.append("promotion review explicitly requested")
    else:
        return {"stage": stage, "reasons": reasons, "blocked_at": "PROMOTION_REVIEW"}

    if reviewed_by:
        stage = "APPROVED"
        reasons.append(f"explicitly approved by {reviewed_by}")
    else:
        return {
            "stage": stage,
            "reasons": reasons + ["no reviewed_by identifier supplied"],
            "blocked_at": "APPROVED",
        }

    if is_shadow_serving:
        stage = "SHADOW"
        reasons.append("actually being served in shadow mode")
    else:
        return {"stage": stage, "reasons": reasons, "blocked_at": "SHADOW"}

    if production_evidence_rows >= MIN_PRODUCTION_EVIDENCE_ROWS:
        stage = "PRODUCTION_VALIDATED"
        reasons.append(f"{production_evidence_rows} production-provenance outcome rows")
    else:
        return {
            "stage": stage,
            "reasons": reasons
            + [
                f"only {production_evidence_rows} production-provenance rows "
                f"(need >= {MIN_PRODUCTION_EVIDENCE_ROWS})"
            ],
            "blocked_at": "PRODUCTION_VALIDATED",
        }

    return {"stage": stage, "reasons": reasons, "blocked_at": None}


def production_validation_status(records: list[PredictionOutcomeRecord]) -> str:
    """PRODUCTION_VALIDATION_STATUS. NOT_READY is an acceptable, expected
    result per the brief - never inflated by counting test/simulated rows."""
    production_rows = [r for r in records if r.provenance == "production"]
    if len(production_rows) >= MIN_PRODUCTION_EVIDENCE_ROWS:
        return "READY"
    return "NOT_READY"


def retraining_readiness(
    *,
    drift_tier: str,
    performance_delta: float | None,
    new_legitimate_labels: int,
    calibration_gap_trend: float | None = None,
) -> dict:
    """RETRAINING_RECOMMENDED / RETRAINING_NOT_RECOMMENDED, from documented
    evidence only. Never triggers retraining itself - the caller (a human, or
    a future scheduled job that still stops at this recommendation) decides
    what to do with it."""
    reasons = []
    recommend = False

    if drift_tier in ("MODERATE_DRIFT", "HIGH_DRIFT"):
        reasons.append(f"feature drift tier is {drift_tier}")
        recommend = True
    if performance_delta is not None and performance_delta > 0.05:
        reasons.append(
            f"mean absolute probability error increased by {performance_delta} "
            "between the first and most recent evaluated period"
        )
        recommend = True
    if calibration_gap_trend is not None and calibration_gap_trend > 0.05:
        reasons.append(f"calibration gap widened by {calibration_gap_trend}")
        recommend = True
    if new_legitimate_labels >= MIN_PRODUCTION_EVIDENCE_ROWS:
        reasons.append(
            f"{new_legitimate_labels} new legitimate labeled outcomes are now available "
            f"(>= the {MIN_PRODUCTION_EVIDENCE_ROWS}-row operational floor)"
        )
        recommend = True

    if not reasons:
        reasons.append("no drift, performance-degradation, calibration, or volume trigger met")

    return {
        "status": "RETRAINING_RECOMMENDED" if recommend else "RETRAINING_NOT_RECOMMENDED",
        "reasons": reasons,
        "auto_retrained": False,  # documents the guarantee - never toggled
        "evaluated_at": datetime.now(UTC).isoformat(),
    }


def data_sufficiency_gate(
    records: list[PredictionOutcomeRecord],
    *,
    min_sample: int = 200,
    min_per_class: int = 30,
) -> dict:
    """RETRAINING_STATUS = DATA_SUFFICIENT / DATA_INSUFFICIENT, from real
    counts on the supplied eligible records - never forces training."""
    n = len(records)
    n_recovered = sum(1 for r in records if r.recovered)
    n_not_recovered = n - n_recovered
    case_ids = {r.case_id for r in records}
    duplicate_contamination = n - len(case_ids)  # >0 would mean the same case counted twice
    legitimate = [r for r in records if r.provenance in ("production", "human_confirmed")]

    checks = {
        "minimum_sample_count": n >= min_sample,
        "class_balance": n_recovered >= min_per_class and n_not_recovered >= min_per_class,
        "outcome_completeness": n
        > 0,  # every record here already has a MATCHED outcome by construction
        "no_duplicate_contamination": duplicate_contamination == 0,
        "legitimate_provenance_present": len(legitimate) > 0,
    }
    failed = [k for k, ok in checks.items() if not ok]

    return {
        "status": "DATA_SUFFICIENT" if not failed else "DATA_INSUFFICIENT",
        "n_total": n,
        "n_recovered": n_recovered,
        "n_not_recovered": n_not_recovered,
        "n_legitimate_provenance": len(legitimate),
        "failed_checks": failed,
        "checks": checks,
    }
