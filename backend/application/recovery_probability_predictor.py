"""The canonical RecoverChain recovery-probability model — Lineage B.

Deliberately separate from application/recovery_predictor_ml.py's
MLPaymentFailurePredictor (Lineage A, payment-failure-risk, inverted). See
docs/ai_reconstruction/MODEL_LINEAGE_RECONCILIATION.md.

STRICT LOAD (ml/inference/model_integrity.py): the newest artifact whose task is
"recovery-probability" is verified (checksum before unpickling, target, feature
count/names/order, preprocessing metadata, the artifact's own internal schema).
If verification fails the model is REFUSED with a deterministic `load_failure`
reason code. There is NO fallback to an older artifact and no fallback to any other
model. Artifacts of other tasks are ignored (a different lineage), never loaded.
"""

import json
import os

from ml.inference.model_integrity import CANONICAL_FEATURES, CANONICAL_TASK, verify_artifact

CANONICAL_RECOVERY_TASK = CANONICAL_TASK
DEFAULT_REGISTRY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "ml", "models", "recovery_registry"
)


class RecoveryProbabilityPredictor:
    def __init__(self, registry_dir: str | None = None):
        self.model = None
        self.metadata: dict = {}
        self.model_id: str | None = None
        self.model_version: str | None = None
        self.calibrated = False  # True = a calibration OBJECT is fitted; NOT outcome-validated
        self.artifact_sha256: str | None = None
        self.load_failure: dict | None = None  # {"reason_code", "detail", "checks_passed"}
        self.integrity_checks: list[str] = []
        reg_dir = registry_dir or DEFAULT_REGISTRY_DIR
        if not os.path.exists(reg_dir):
            self.load_failure = {
                "reason_code": "REGISTRY_MISSING",
                "detail": reg_dir,
                "checks_passed": [],
            }
            return

        candidates = []
        for f in os.listdir(reg_dir):
            if not f.endswith("_metadata.json"):
                continue
            try:
                with open(os.path.join(reg_dir, f)) as fh:
                    meta = json.load(fh)
            except Exception:
                continue
            if meta.get("task") == CANONICAL_TASK:  # other tasks belong to other lineages
                candidates.append((meta.get("created_at") or "", f))
        if not candidates:
            self.load_failure = {
                "reason_code": "NO_CANONICAL_ARTIFACT",
                "detail": reg_dir,
                "checks_passed": [],
            }
            return

        newest = max(candidates)[1]  # only the newest is considered: no fallback to older ones
        result = verify_artifact(reg_dir, os.path.join(reg_dir, newest))
        if not result.ok:
            self.load_failure = {
                "reason_code": result.reason_code,
                "detail": result.detail,
                "checks_passed": result.checks_passed,
            }
            return
        self.model = result.model
        self.metadata = result.metadata
        self.model_id = result.metadata.get("model_id")
        self.model_version = result.metadata.get("selected_model")
        self.calibrated = bool(result.metadata.get("calibrated"))
        self.artifact_sha256 = result.artifact_sha256
        self.integrity_checks = result.checks_passed

    def predict(self, features: dict) -> dict:
        """`features` must contain exactly the canonical feature names (values may be None =
        missing, handled by the trained imputer). Columns are assembled in the registered
        schema order. Extra/missing keys are refused."""
        if self.model is None:
            return {"status": "NO_MODEL", "probability": None, "load_failure": self.load_failure}
        if set(features) != set(CANONICAL_FEATURES):
            return {
                "status": "ERROR",
                "probability": None,
                "error": "FEATURE_SCHEMA_MISMATCH: "
                f"missing={sorted(set(CANONICAL_FEATURES) - set(features))} "
                f"extra={sorted(set(features) - set(CANONICAL_FEATURES))}",
            }
        try:
            import pandas as pd

            row = pd.DataFrame([{k: features[k] for k in CANONICAL_FEATURES}])
            prob = float(self.model.predict_proba(row)[0][1])
            raw = None
            try:
                base = self.model.calibrated_classifiers_[0].estimator
                base = getattr(base, "estimator", base)
                raw = float(base.predict_proba(row)[0][1])
            except Exception:
                raw = None
        except Exception as e:
            return {"status": "ERROR", "probability": None, "error": str(e)}
        return {
            "status": "SUCCESS",
            "probability": max(0.0, min(1.0, prob)),
            "raw_probability": None if raw is None else max(0.0, min(1.0, raw)),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "calibrated": self.calibrated,
            "artifact_sha256": self.artifact_sha256,
        }
