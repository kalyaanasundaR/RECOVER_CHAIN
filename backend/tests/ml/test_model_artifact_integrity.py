"""Strict load-time integrity for the canonical recovery model (ml/inference/model_integrity.py).
Uses temporary artifacts; the production artifact is never mutated."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from application.recovery_probability_predictor import RecoveryProbabilityPredictor
from ml.inference.model_integrity import (
    CANONICAL_FEATURES,
    build_feature_schema,
    sha256_file,
    verify_artifact,
)

BACKEND = Path(__file__).resolve().parents[2]


def _edit_meta(meta_path, fn):
    m = json.loads(Path(meta_path).read_text())
    fn(m)
    Path(meta_path).write_text(json.dumps(m))


def test_valid_artifact_passes_every_check(make_recovery_artifact):
    d, m, a = make_recovery_artifact()
    r = verify_artifact(str(d), str(m))
    assert r.ok and r.reason_code == "OK"
    expected = {"task", "target", "checksum", "feature_schema", "preprocessing_metadata"}
    assert expected <= set(r.checks_passed)
    assert "artifact_internal_schema" in r.checks_passed
    assert r.artifact_sha256 == sha256_file(a)
    p = RecoveryProbabilityPredictor(registry_dir=str(d))
    assert p.model is not None and p.load_failure is None and p.artifact_sha256 == r.artifact_sha256


def test_modified_artifact_is_refused_before_unpickling(make_recovery_artifact):
    d, m, a = make_recovery_artifact()
    with open(a, "ab") as f:
        f.write(b"\x00tampered")
    r = verify_artifact(str(d), str(m))
    assert not r.ok and r.reason_code == "CHECKSUM_MISMATCH" and r.model is None
    p = RecoveryProbabilityPredictor(registry_dir=str(d))
    assert p.model is None and p.load_failure["reason_code"] == "CHECKSUM_MISMATCH"


def test_wrong_registered_checksum_is_refused(make_recovery_artifact):
    d, m, _ = make_recovery_artifact()
    _edit_meta(m, lambda x: x.update(artifact_sha256="0" * 64))
    assert verify_artifact(str(d), str(m)).reason_code == "CHECKSUM_MISMATCH"


def test_missing_checksum_is_refused_not_defaulted(make_recovery_artifact):
    d, m, _ = make_recovery_artifact(register=False)
    assert verify_artifact(str(d), str(m)).reason_code == "CHECKSUM_MISSING"
    assert RecoveryProbabilityPredictor(registry_dir=str(d)).model is None


def test_wrong_task(make_recovery_artifact):
    d, m, _ = make_recovery_artifact()
    _edit_meta(m, lambda x: x.update(task="payment-failure-risk"))
    assert verify_artifact(str(d), str(m)).reason_code == "WRONG_TASK"
    # a differently-tasked artifact is another lineage: ignored, never loaded here
    p = RecoveryProbabilityPredictor(registry_dir=str(d))
    assert p.model is None and p.load_failure["reason_code"] == "NO_CANONICAL_ARTIFACT"


def test_wrong_target_and_target_definition(make_recovery_artifact):
    d, m, _ = make_recovery_artifact()
    _edit_meta(m, lambda x: x.update(target_column="target_late"))
    assert verify_artifact(str(d), str(m)).reason_code == "WRONG_TARGET"
    _edit_meta(
        m, lambda x: x.update(target_column="recovery_success", target_definition="1 = paid late")
    )
    assert verify_artifact(str(d), str(m)).reason_code == "TARGET_DEFINITION_MISMATCH"


def test_missing_feature_schema(make_recovery_artifact):
    d, m, _ = make_recovery_artifact()
    _edit_meta(m, lambda x: x.pop("feature_schema"))
    assert verify_artifact(str(d), str(m)).reason_code == "FEATURE_SCHEMA_MISSING"


def test_missing_feature_gives_count_mismatch(make_recovery_artifact):
    d, m, _ = make_recovery_artifact()
    _edit_meta(m, lambda x: x["feature_schema"]["names"].pop())
    assert verify_artifact(str(d), str(m)).reason_code == "FEATURE_COUNT_MISMATCH"


def test_extra_feature_gives_count_mismatch(make_recovery_artifact):
    d, m, _ = make_recovery_artifact()
    _edit_meta(m, lambda x: x["feature_schema"]["names"].append("bonus_feature"))
    assert verify_artifact(str(d), str(m)).reason_code == "FEATURE_COUNT_MISMATCH"


def test_renamed_feature_gives_names_mismatch(make_recovery_artifact):
    d, m, _ = make_recovery_artifact()
    _edit_meta(m, lambda x: x["feature_schema"]["names"].__setitem__(0, "amount_at_risk_v2"))
    r = verify_artifact(str(d), str(m))
    assert r.reason_code == "FEATURE_NAMES_MISMATCH" and "amount_at_risk_v2" in r.detail


def test_reordered_feature_schema(make_recovery_artifact):
    d, m, _ = make_recovery_artifact()
    _edit_meta(m, lambda x: x["feature_schema"]["names"].reverse())
    assert verify_artifact(str(d), str(m)).reason_code == "FEATURE_ORDER_MISMATCH"


def test_incompatible_preprocessing_metadata(make_recovery_artifact):
    d, m, _ = make_recovery_artifact()
    _edit_meta(m, lambda x: x["preprocessing"]["numeric_columns"].reverse())
    assert verify_artifact(str(d), str(m)).reason_code == "PREPROCESSING_METADATA_MISMATCH"


def test_artifact_whose_internal_pipeline_disagrees_with_its_metadata(make_recovery_artifact):
    # Trained with a reversed column order, but its metadata claims the canonical schema.
    d, m, a = make_recovery_artifact(
        column_order=list(reversed(CANONICAL_FEATURES)), register=False
    )
    kinds = build_feature_schema()["kinds"]
    _edit_meta(
        m,
        lambda x: x.update(
            artifact_sha256=sha256_file(a),
            feature_schema=build_feature_schema(),
            preprocessing={
                "numeric_columns": [c for c in CANONICAL_FEATURES if kinds[c] == "numeric"],
                "categorical_columns": [c for c in CANONICAL_FEATURES if kinds[c] == "categorical"],
            },
        ),
    )
    assert verify_artifact(str(d), str(m)).reason_code == "ARTIFACT_INTERNAL_SCHEMA_MISMATCH"


def test_missing_artifact_file(make_recovery_artifact):
    d, m, a = make_recovery_artifact()
    a.unlink()
    assert verify_artifact(str(d), str(m)).reason_code == "ARTIFACT_MISSING"


def test_registry_inconsistent_model_id(make_recovery_artifact):
    d, m, _ = make_recovery_artifact()
    _edit_meta(m, lambda x: x.update(model_id="recovery_model_other"))
    assert verify_artifact(str(d), str(m)).reason_code == "REGISTRY_INCONSISTENT"


def test_no_silent_fallback_to_an_older_valid_artifact(make_recovery_artifact):
    d, m_old, _ = make_recovery_artifact("recovery_model_old", created="2026-01-01")
    _, _, a_new = make_recovery_artifact("recovery_model_new", created="2026-02-01")
    with open(a_new, "ab") as f:
        f.write(b"x")  # the NEWEST artifact is corrupted; the older one is valid
    assert verify_artifact(str(d), str(m_old)).ok
    p = RecoveryProbabilityPredictor(registry_dir=str(d))
    assert p.model is None and p.load_failure["reason_code"] == "CHECKSUM_MISMATCH"


def test_predict_refuses_missing_and_extra_keys_and_uses_schema_order(make_recovery_artifact):
    d, _, _ = make_recovery_artifact()
    p = RecoveryProbabilityPredictor(registry_dir=str(d))
    vec = dict.fromkeys(CANONICAL_FEATURES)
    vec["amount_at_risk"] = 1.0
    ok = p.predict(dict(reversed(list(vec.items()))))  # dict order must not matter
    assert ok["status"] == "SUCCESS" and ok["artifact_sha256"] == p.artifact_sha256
    assert p.predict({k: v for k, v in vec.items() if k != "domain"})["status"] == "ERROR"
    assert p.predict({**vec, "surprise": 1})["status"] == "ERROR"


def test_restart_load_verification_in_a_fresh_process(make_recovery_artifact):
    d, _, a = make_recovery_artifact()
    code = (
        "import sys; sys.path.insert(0, '.');"
        "from application.recovery_probability_predictor import RecoveryProbabilityPredictor as P;"
        f"p=P(registry_dir=r'{d}'); print(p.model_id, p.artifact_sha256, p.load_failure)"
    )
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], cwd=BACKEND, capture_output=True, text=True
    )
    assert out.returncode == 0, out.stderr[-500:]
    assert f"recovery_model_test1 {sha256_file(a)} None" in out.stdout


def test_production_artifact_is_registered_and_matches_an_independent_hash():
    from application.recovery_probability_predictor import DEFAULT_REGISTRY_DIR

    if not list(Path(DEFAULT_REGISTRY_DIR).glob("*_metadata.json")):
        pytest.skip("no production canonical artifact present")
    p = RecoveryProbabilityPredictor()
    assert p.model is not None, p.load_failure
    path = Path(DEFAULT_REGISTRY_DIR, f"{p.model_id}_model.joblib")
    assert p.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
