"""Read-only breakdown of the two probability lineages on REAL cases in the dev database.

Lineage A: the stored case.prediction (mostly the deterministic BASELINE; sometimes the
per-upload trained shadow model). Lineage B: the canonical recovery model, only when it
passes integrity verification. This is a disagreement/coverage inventory, NOT a benchmark:
neither model is ranked and no accuracy-type metric is computed - no outcome labels exist
for the compared cases (LIVE_OUTCOME_EVALUATION = DATA_BLOCKED).

Usage: venv/Scripts/python.exe ml_reconstruction/lineage_disagreement.py [max_cases]
"""

import json
import sys
from collections import Counter
from datetime import UTC, datetime

import numpy as np

sys.path.insert(0, ".")
from application.recovery_probability_predictor import RecoveryProbabilityPredictor  # noqa: E402
from infrastructure.db import SessionLocal  # noqa: E402
from infrastructure.orm import CaseModel, OutcomeEventModel  # noqa: E402
from ml.inference.recovery_features import derive_recovery_features  # noqa: E402


def _stats(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {"n": int(len(a))}
    if len(a):
        out["mean_abs_difference"] = float(np.mean(np.abs(a - b)))
        out["current_mean"] = float(a.mean())
        out["canonical_mean"] = float(b.mean())
    if len(a) >= 10 and a.std() > 0 and b.std() > 0:
        out["pearson_correlation"] = float(np.corrcoef(a, b)[0, 1])
    return out


def main(limit=300):
    db = SessionLocal()
    pred = RecoveryProbabilityPredictor()
    now = datetime.now(UTC)
    rows = []
    for c in db.query(CaseModel).filter(CaseModel.prediction.isnot(None)).limit(limit):
        p = c.prediction or {}
        if p.get("recovery_probability") is None:
            continue
        rec = {
            "case_id": c.case_id,
            "current_p": float(p["recovery_probability"]),
            "current_source": p.get("model_version", "?"),
            "canonical_p": None,
            "coverage": None,
        }
        if pred.model is not None:
            fv = derive_recovery_features(db, c.case_id, now)
            r = pred.predict(fv.values)
            rec["coverage"] = fv.available_count
            if r["status"] == "SUCCESS":
                rec["canonical_p"] = r["probability"]
        rows.append(rec)

    is_baseline = lambda r: str(r["current_source"]).startswith("baseline")  # noqa: E731
    A = [r for r in rows if is_baseline(r)]
    B = [r for r in rows if not is_baseline(r)]
    C = [r for r in rows if r["canonical_p"] is not None]
    cov = [r["coverage"] for r in rows if r["coverage"] is not None]

    # Outcomes: applied settlement events matched to these cases, by provenance.
    ids = {r["case_id"] for r in rows}
    ev = (
        db.query(OutcomeEventModel)
        .filter(OutcomeEventModel.case_id.in_(ids), OutcomeEventModel.effect == "APPLIED")
        .all()
        if ids
        else []
    )
    with_outcome = {e.case_id for e in ev}
    prov = Counter()
    for e in ev:
        prov[e.provenance] += 1
    production = sum(1 for e in ev if e.provenance == "production")

    def both(g):
        g = [r for r in g if r["canonical_p"] is not None]
        return _stats([r["current_p"] for r in g], [r["canonical_p"] for r in g])

    out = {
        "cases_examined": len(rows),
        "canonical_model_integrity": "VERIFIED"
        if pred.model is not None
        else f"REFUSED {pred.load_failure}",
        "A_current_probability_is_deterministic_baseline": len(A),
        "B_current_probability_is_trained_shadow_model": len(B),
        "C_canonical_prediction_available": len(C),
        "D_feature_coverage_distribution_available_of_23": dict(sorted(Counter(cov).items())),
        "D_note": "no adequacy threshold is defined (business owner decides); distribution only. "
        "Theoretical maximum today is 14/23.",
        "E_cases_with_applied_settlement_outcome": len(with_outcome),
        "E_outcome_events_by_provenance": dict(prov),
        "E_production_provenance_outcomes": production,
        "F_cases_without_any_outcome": len(ids - with_outcome),
        "prediction_disagreement_only_not_a_benchmark": {
            "A_baseline_vs_canonical": both(A),
            "B_trained_shadow_vs_canonical": both(B),
            "all_vs_canonical": both(rows),
        },
        "LIVE_OUTCOME_EVALUATION": "DATA_BLOCKED"
        if production == 0
        else "NOT_COMPUTED_HERE (production outcomes exist; needs the ERV_MIGRATION_CRITERIA protocol)",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
