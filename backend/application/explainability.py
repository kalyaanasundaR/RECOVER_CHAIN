"""Phase 12 — explainable AI for recovery-probability predictions.

No SHAP library is installed in this environment (verified: `import shap` fails).
Per the brief's own instruction ("otherwise use an appropriate model-specific or
perturbation-based explanation"), this module uses two genuine, dependency-free
techniques instead of installing a new dependency mid-hardening-pass:

  - Legacy XGBoost Booster: `Booster.predict(..., pred_contribs=True)` — XGBoost's
    OWN exact, additive per-feature contribution decomposition (the same
    mathematical object SHAP's TreeExplainer produces for tree ensembles,
    computed natively, no extra package).
  - Everything else (the live sklearn Pipeline / CalibratedClassifierCV models
    MLTrainingEngine produces): leave-one-out perturbation. Each feature is, in
    turn, replaced with a missing value; because the pipeline's own
    ColumnTransformer already imputes missing values with the TRAINING median
    (numeric) or a constant "missing" category (categorical) - see
    application/ml_training.py's `numeric_transformer`/`categorical_transformer`
    - re-running predict_proba with a feature "removed" this way asks the model
    "what would you have predicted using only your own learned default for this
    feature", which is a real, honest measurement of that feature's marginal
    contribution to THIS prediction, not an invented number.

Both paths produce the same output shape: a list of
{feature, value, contribution, direction} sorted by |contribution| descending,
where `direction` is "increased" or "decreased" relative to the model's mean/
default prediction for that feature — i.e. a MODEL ASSOCIATION, never a claim
about real-world causation. Nothing here fabricates a causal explanation from
free text; every number traces back to an actual model call on actual input.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TOP_K_FEATURES = 5


def _direction(contribution: float) -> str:
    if contribution > 1e-9:
        return "increased"
    if contribution < -1e-9:
        return "decreased"
    return "no_effect"


def explain_sklearn_pipeline(
    model: Any, features: list[str], mapped_features: dict, top_k: int = TOP_K_FEATURES
) -> list[dict]:
    """Leave-one-out perturbation explanation for the live (non-legacy) model
    path. `model` is whatever MLPaymentFailurePredictor.model holds (a Pipeline
    or a CalibratedClassifierCV wrapping one) - both expose predict_proba."""
    base_df = pd.DataFrame([{f: mapped_features.get(f) for f in features}])
    base_prob = float(model.predict_proba(base_df[features])[0, 1])

    contributions = []
    for f in features:
        perturbed = base_df.copy()
        perturbed.at[0, f] = None  # falls through to the pipeline's own imputer
        try:
            perturbed_prob = float(model.predict_proba(perturbed[features])[0, 1])
        except Exception:
            # A feature whose absence the fitted preprocessor can't handle
            # (e.g. all-categorical OneHotEncoder edge case) is reported as
            # UNKNOWN rather than silently assigned zero contribution.
            contributions.append(
                {
                    "feature": f,
                    "value": mapped_features.get(f),
                    "contribution": None,
                    "direction": "UNKNOWN",
                }
            )
            continue
        contribution = base_prob - perturbed_prob
        contributions.append(
            {
                "feature": f,
                "value": mapped_features.get(f),
                "contribution": round(contribution, 6),
                "direction": _direction(contribution),
            }
        )

    contributions.sort(key=lambda c: abs(c["contribution"] or 0.0), reverse=True)
    return contributions[:top_k]


def explain_legacy_xgboost(
    booster, features: list[str], mapped_features: dict, top_k: int = TOP_K_FEATURES
) -> list[dict]:
    """Exact per-feature contributions from XGBoost's own additive decomposition
    (`pred_contribs=True`) - the model's raw output is FAILURE risk, so a
    positive contribution here increased failure risk, i.e. DECREASED recovery
    probability; the sign is flipped below so `direction` stays in recovery-
    probability terms, consistent with explain_sklearn_pipeline's output."""
    import xgboost as xgb

    x = [mapped_features.get(f, 0.0) for f in features]
    dmat = xgb.DMatrix(np.array([x]), feature_names=features)
    contribs = booster.predict(dmat, pred_contribs=True)[0]  # last entry is the bias term

    contributions = []
    for i, f in enumerate(features):
        recovery_contribution = -float(contribs[i])  # flip: failure-risk -> recovery-probability
        contributions.append(
            {
                "feature": f,
                "value": mapped_features.get(f),
                "contribution": round(recovery_contribution, 6),
                "direction": _direction(recovery_contribution),
            }
        )

    contributions.sort(key=lambda c: abs(c["contribution"] or 0.0), reverse=True)
    return contributions[:top_k]


def confidence_tier(probability: float, model_metadata: dict) -> str:
    """Honest, documented-methodology confidence bucket - NOT a calibrated
    statistical confidence interval (this repo's calibration was shown in
    Phase 10 to be inconclusive at n=429; fabricating a CI on top of that would
    misrepresent it). Methodology, stated plainly so it can be audited:

      LOW    - the model's own training-time quality gate flagged small_sample,
               OR the registry has no quality_gate info at all, OR the
               prediction sits within 10 points of the decision boundary
               (0.40-0.60) where a small probability error changes the
               practical action most.
      MEDIUM - decision margin is wider (outside 0.40-0.60) but the model is
               small_sample or otherwise not fully quality-vetted.
      HIGH   - decision margin is wide AND the model passed its quality gate
               at full (non-adaptive-minimum) sample size.

    This is a margin+provenance heuristic, not a probabilistic guarantee -
    documented as such everywhere it is surfaced."""
    quality_gate = (model_metadata or {}).get("quality_gate") or {}
    small_sample = quality_gate.get("small_sample")
    gate_passed = quality_gate.get("passed")
    near_boundary = 0.40 <= probability <= 0.60

    if small_sample is None or gate_passed is None:
        return "LOW"  # no quality-gate evidence at all - can't honestly claim more
    if near_boundary:
        return "LOW"  # a small probability error changes the practical action most here
    if small_sample or not gate_passed:
        return "MEDIUM"
    return "HIGH"


def to_human_readable(explanation: list[dict], probability: float) -> list[str]:
    """Plain-language bullets from ACTUAL contributions - never invents a
    factor that isn't in `explanation`. Every line is a MODEL ASSOCIATION
    statement (what correlates with this model's output), explicitly not a
    BUSINESS CAUSATION claim (what caused the customer's behaviour)."""
    lines = []
    for c in explanation:
        if c["direction"] == "UNKNOWN" or c["contribution"] is None:
            continue
        verb = "increased" if c["direction"] == "increased" else "decreased"
        magnitude = abs(c["contribution"])
        if magnitude < 0.005:
            continue  # not worth surfacing - rounds to ~0% at 2dp display
        lines.append(
            f"{c['feature']}={c['value']!r} is associated with a "
            f"{verb} model-estimated recovery probability "
            f"(MODEL ASSOCIATION, not a claim of business causation)."
        )
    if not lines:
        lines.append(
            "No single feature had a measurable effect on this prediction "
            "(MODEL ASSOCIATION analysis found no dominant factor)."
        )
    return lines
