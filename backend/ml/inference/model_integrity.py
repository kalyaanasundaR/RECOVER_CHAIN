"""Strict load-time integrity verification for the canonical recovery-probability artifact.

`verify_artifact(registry_dir, metadata_path)` returns an `IntegrityResult`. Any critical
failure gives a deterministic `reason_code`; the caller must REFUSE to serve the model
(no fallback to another artifact, no silent tolerance). The artifact is checksummed
BEFORE it is unpickled, so a tampered file is never deserialised.

Registered metadata (written by `register_artifact` / the training script) must carry:
  artifact_sha256, feature_schema{names, kinds, schema_sha256}, preprocessing{...},
  categorical_vocabulary, target_column, target_definition, task, model_id, ...
The registered checksum is computed from the real file bytes — never invented.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

CANONICAL_TASK = "recovery-probability"
CANONICAL_TARGET = "recovery_success"
CANONICAL_TARGET_DEFINITION = (
    "1 = at-risk amount recovered within the dataset's 30-day observation window"
)
ALLOWED_MODELS = {"logistic_regression", "random_forest", "xgboost"}

# Ordered exactly as ml/training/train_recovery_model.py FEATURES (NUMERIC + CATEGORICAL).
CANONICAL_NUMERIC = [
    "amount_at_risk",
    "age_of_obligation",
    "customer_tenure",
    "prior_payment_success_rate",
    "prior_recovery_rate",
    "prior_late_rate",
    "number_of_previous_failed_attempts",
    "number_of_previous_recoveries",
    "average_days_to_payment",
    "amount_vs_customer_average",
    "invoice_age",
    "payment_method_valid",
    "customer_engagement",
    "recent_login",
    "reminder_history",
    "previous_contact_success",
    "dispute_indicator",
    "suppression_status",
]
CANONICAL_CATEGORICAL = [
    "domain",
    "payment_method",
    "account_status",
    "consent_status",
    "historical_recovery_behavior",
]
CANONICAL_FEATURES = CANONICAL_NUMERIC + CANONICAL_CATEGORICAL


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def schema_sha256(names: list[str], kinds: dict[str, str]) -> str:
    payload = json.dumps({"names": names, "kinds": kinds}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def build_feature_schema() -> dict:
    kinds = {n: "numeric" for n in CANONICAL_NUMERIC} | {
        n: "categorical" for n in CANONICAL_CATEGORICAL
    }
    return {
        "names": list(CANONICAL_FEATURES),
        "kinds": kinds,
        "schema_sha256": schema_sha256(list(CANONICAL_FEATURES), kinds),
    }


@dataclass
class IntegrityResult:
    ok: bool
    reason_code: str = "OK"
    detail: str = ""
    checks_passed: list[str] = field(default_factory=list)
    artifact_sha256: str | None = None
    model: object | None = None
    metadata: dict = field(default_factory=dict)


def _fail(code, detail, passed, meta=None, sha=None):
    return IntegrityResult(False, code, detail, list(passed), sha, None, meta or {})


def _base_pipeline(model):
    """Frozen base pipeline inside a CalibratedClassifierCV (or the pipeline itself)."""
    try:
        est = model.calibrated_classifiers_[0].estimator
        return getattr(est, "estimator", est)
    except Exception:
        return model


def verify_artifact(registry_dir: str, metadata_path: str) -> IntegrityResult:
    passed: list[str] = []
    try:
        with open(metadata_path) as f:
            meta = json.load(f)
    except Exception as e:
        return _fail("METADATA_UNREADABLE", str(e), passed)

    # -- 3/4/10: task, target, registry consistency ---------------------------------
    if meta.get("task") != CANONICAL_TASK:
        return _fail("WRONG_TASK", f"task={meta.get('task')!r}", passed, meta)
    passed.append("task")
    if meta.get("target_column") != CANONICAL_TARGET:
        return _fail("WRONG_TARGET", f"target_column={meta.get('target_column')!r}", passed, meta)
    if meta.get("target_definition") != CANONICAL_TARGET_DEFINITION:
        return _fail("TARGET_DEFINITION_MISMATCH", str(meta.get("target_definition")), passed, meta)
    passed.append("target")

    model_id = meta.get("model_id")
    if not model_id or not Path(metadata_path).name.startswith(f"{model_id}_"):
        return _fail(
            "REGISTRY_INCONSISTENT", "model_id does not match metadata file name", passed, meta
        )
    if meta.get("status") != "SELECTED":
        return _fail("STATUS_NOT_SELECTED", f"status={meta.get('status')!r}", passed, meta)
    if meta.get("selected_model") not in ALLOWED_MODELS:
        return _fail("UNKNOWN_MODEL_TYPE", str(meta.get("selected_model")), passed, meta)
    stored_path = meta.get("artifact_path")
    expected_name = f"{model_id}_model.joblib"
    if stored_path and Path(stored_path).name != expected_name:
        return _fail(
            "REGISTRY_INCONSISTENT",
            "artifact_path file name != <model_id>_model.joblib",
            passed,
            meta,
        )
    passed.append("registry_consistency")

    # -- 1: artifact exists (resolved from the registry dir, not a stored absolute path) --
    artifact = Path(registry_dir) / expected_name
    if not artifact.exists():
        return _fail("ARTIFACT_MISSING", str(artifact), passed, meta)
    passed.append("artifact_exists")

    # -- 2: checksum BEFORE unpickling ------------------------------------------------
    registered = meta.get("artifact_sha256")
    if not registered:
        return _fail("CHECKSUM_MISSING", "no artifact_sha256 registered", passed, meta)
    actual = sha256_file(artifact)
    if registered != actual:
        return _fail(
            "CHECKSUM_MISMATCH",
            f"registered={registered[:12]} actual={actual[:12]}",
            passed,
            meta,
            actual,
        )
    passed.append("checksum")

    # -- 5/6/7/8: feature schema ----------------------------------------------------
    schema = meta.get("feature_schema")
    if not schema or "names" not in schema:
        return _fail(
            "FEATURE_SCHEMA_MISSING", "metadata has no feature_schema", passed, meta, actual
        )
    names = schema["names"]
    if len(names) != len(CANONICAL_FEATURES):
        return _fail(
            "FEATURE_COUNT_MISMATCH",
            f"{len(names)} != {len(CANONICAL_FEATURES)}",
            passed,
            meta,
            actual,
        )
    if set(names) != set(CANONICAL_FEATURES):
        missing = sorted(set(CANONICAL_FEATURES) - set(names))
        extra = sorted(set(names) - set(CANONICAL_FEATURES))
        return _fail(
            "FEATURE_NAMES_MISMATCH", f"missing={missing} extra={extra}", passed, meta, actual
        )
    if names != CANONICAL_FEATURES:
        return _fail("FEATURE_ORDER_MISMATCH", "same names, different order", passed, meta, actual)
    if (
        meta.get("feature_columns") is not None
        and list(meta["feature_columns"]) != CANONICAL_FEATURES
    ):
        return _fail(
            "FEATURE_SCHEMA_INCONSISTENT",
            "feature_columns != feature_schema.names",
            passed,
            meta,
            actual,
        )
    if schema.get("schema_sha256") != build_feature_schema()["schema_sha256"]:
        return _fail(
            "FEATURE_SCHEMA_HASH_MISMATCH",
            "kinds/names hash differs from the live contract",
            passed,
            meta,
            actual,
        )
    passed.append("feature_schema")

    # -- 9: preprocessing metadata ----------------------------------------------------
    pre = meta.get("preprocessing")
    if (
        not pre
        or pre.get("numeric_columns") != CANONICAL_NUMERIC
        or pre.get("categorical_columns") != CANONICAL_CATEGORICAL
    ):
        return _fail(
            "PREPROCESSING_METADATA_MISMATCH",
            "numeric/categorical column lists differ",
            passed,
            meta,
            actual,
        )
    passed.append("preprocessing_metadata")

    # -- only now deserialise; then verify the artifact's OWN internal schema -------------
    try:
        import joblib

        model = joblib.load(artifact)
    except Exception as e:
        return _fail("ARTIFACT_UNLOADABLE", str(e), passed, meta, actual)
    try:
        base = _base_pipeline(model)
        ct = base.named_steps["pre"]
        num_cols = list(next(t for t in ct.transformers_ if t[0] == "num")[2])
        cat_cols = list(next(t for t in ct.transformers_ if t[0] == "cat")[2])
        if list(ct.feature_names_in_) != CANONICAL_FEATURES:
            return _fail(
                "ARTIFACT_INTERNAL_SCHEMA_MISMATCH",
                "pipeline feature_names_in_ differs",
                passed,
                meta,
                actual,
            )
        if num_cols != CANONICAL_NUMERIC or cat_cols != CANONICAL_CATEGORICAL:
            return _fail(
                "ARTIFACT_INTERNAL_SCHEMA_MISMATCH",
                "transformer columns differ",
                passed,
                meta,
                actual,
            )
        vocab = meta.get("categorical_vocabulary")
        if vocab is not None:
            enc = next(t for t in ct.transformers_ if t[0] == "cat")[1]
            actual_vocab = {
                c: [str(x) for x in cats] for c, cats in zip(cat_cols, enc.categories_, strict=True)
            }
            if actual_vocab != vocab:
                return _fail(
                    "CATEGORY_VOCABULARY_MISMATCH",
                    "encoder categories differ from registered",
                    passed,
                    meta,
                    actual,
                )
    except Exception as e:
        return _fail("ARTIFACT_INTERNAL_SCHEMA_UNREADABLE", str(e), passed, meta, actual)
    passed.append("artifact_internal_schema")

    if bool(meta.get("calibrated")) != hasattr(model, "calibrated_classifiers_"):
        return _fail(
            "CALIBRATION_METADATA_MISMATCH",
            "calibrated flag != artifact type",
            passed,
            meta,
            actual,
        )
    passed.append("calibration_object_consistency")

    return IntegrityResult(True, "OK", "all checks passed", passed, actual, model, meta)


def register_artifact(registry_dir: str, metadata_path: str) -> dict:
    """Compute and write the integrity fields into an existing artifact's metadata from the
    ACTUAL artifact bytes and its ACTUAL internal pipeline. Does not touch the model file."""
    import joblib

    with open(metadata_path) as f:
        meta = json.load(f)
    artifact = Path(registry_dir) / f"{meta['model_id']}_model.joblib"
    model = joblib.load(artifact)
    ct = _base_pipeline(model).named_steps["pre"]
    enc = next(t for t in ct.transformers_ if t[0] == "cat")[1]
    cat_cols = list(next(t for t in ct.transformers_ if t[0] == "cat")[2])
    num_cols = list(next(t for t in ct.transformers_ if t[0] == "num")[2])
    meta["artifact_sha256"] = sha256_file(artifact)
    meta["feature_schema"] = build_feature_schema()
    meta["preprocessing"] = {
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "numeric_steps": "SimpleImputer(median) -> StandardScaler",
        "categorical_step": "OneHotEncoder(handle_unknown=ignore)",
        "missing_values": "NaN/None passed to the trained imputer; never fabricated",
    }
    meta["categorical_vocabulary"] = {
        c: [str(x) for x in cats] for c, cats in zip(cat_cols, enc.categories_, strict=True)
    }
    meta["integrity_registered"] = True
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta
