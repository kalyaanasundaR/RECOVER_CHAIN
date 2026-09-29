"""Phase 12 — lightweight anomaly detection, separate from recovery probability.

Answers one real question: "is this revenue event materially different from the
distribution the shadow model was actually trained on?" — not "how likely is
recovery" (that's recovery_predictor_ml.py) and not a general-purpose outlier
detector bolted on to raise the technology count.

Approach: a statistical (z-score) check against the per-feature training
distribution already computed by application.drift_readiness.summarize_distribution
and persisted in registry metadata as `training_feature_distribution`
(application/ml_training.py). This was chosen over sklearn's IsolationForest for
two concrete reasons, not by default:
  1. It requires no additional persisted artifact or fit-time cost - the same
     summary drift_readiness already computes and stores answers this too.
  2. It is directly interpretable per-feature ("amount is 4.2 std above the
     training mean"), which is what an operator needs to trust or dismiss the
     flag; IsolationForest's anomaly score has no such per-feature attribution
     without a second explainability pass on top of it.
This is a documented trade-off, not an oversight - Isolation Forest remains a
reasonable alternative if a future phase needs multivariate (not per-feature)
anomaly structure.
"""

from __future__ import annotations

UNUSUAL_Z = 2.0
ANOMALOUS_Z = 3.5


def score_anomaly(feature_values: dict, training_distribution: dict | None) -> dict:
    """Returns {"level": NORMAL|UNUSUAL|ANOMALOUS|UNKNOWN, "max_abs_z": float|None,
    "per_feature": {feature: z_score}}. UNKNOWN (not NORMAL) when no training
    distribution is available - absence of evidence is not evidence of
    normality."""
    if not training_distribution:
        return {"level": "UNKNOWN", "max_abs_z": None, "per_feature": {}}

    per_feature: dict[str, float] = {}
    for feature, summary in training_distribution.items():
        if not isinstance(summary, dict) or summary.get("type") != "numeric":
            continue
        if feature not in feature_values or feature_values[feature] is None:
            continue
        std = summary.get("std") or 0.0
        if std == 0:
            continue
        try:
            value = float(feature_values[feature])
        except (TypeError, ValueError):
            continue
        per_feature[feature] = round((value - summary["mean"]) / std, 3)

    if not per_feature:
        return {"level": "UNKNOWN", "max_abs_z": None, "per_feature": {}}

    max_abs_z = max(abs(z) for z in per_feature.values())
    if max_abs_z >= ANOMALOUS_Z:
        level = "ANOMALOUS"
    elif max_abs_z >= UNUSUAL_Z:
        level = "UNUSUAL"
    else:
        level = "NORMAL"

    return {"level": level, "max_abs_z": round(max_abs_z, 3), "per_feature": per_feature}
