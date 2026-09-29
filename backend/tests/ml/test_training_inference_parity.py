"""Training/inference parity for the canonical recovery model. These fail loudly if the
training definition and the live derivation drift apart (names, order, dtypes, vocabulary,
preprocessing, rolling-history semantics, cutoff, cold start, missing handling)."""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from application.recovery_probability_predictor import (
    DEFAULT_REGISTRY_DIR,
    RecoveryProbabilityPredictor,
)
from domain.models import RiskCategory
from infrastructure.db import SessionLocal
from infrastructure.orm import (
    AuditModel,
    CaseModel,
    ContactPermissionModel,
    EventModel,
    OutcomeEventModel,
)
from ml.inference.model_integrity import (
    CANONICAL_CATEGORICAL,
    CANONICAL_FEATURES,
    CANONICAL_NUMERIC,
    _base_pipeline,
)
from ml.inference.recovery_features import (
    COLD_START_PRIOR_RECOVERY_RATE,
    TRAINED_CONSENT_STATUSES,
    TRAINED_PAYMENT_METHODS,
    derive_recovery_features,
)
from ml.training import train_recovery_model as training

DATA = (
    Path(__file__).resolve().parents[3]
    / "data/recoverchain/synthetic/recovery_outcome/recovery_outcome.csv"
)
T0 = datetime(2026, 3, 1, 9, 0, 0)


@pytest.fixture(scope="module")
def prod():
    p = RecoveryProbabilityPredictor()
    if p.model is None:
        pytest.skip("no production canonical artifact present")
    return p


def test_feature_names_order_and_kinds_match_across_training_contract_metadata_and_artifact(prod):
    assert training.FEATURES == CANONICAL_FEATURES  # training script == inference contract
    assert training.NUMERIC == CANONICAL_NUMERIC and training.CATEGORICAL == CANONICAL_CATEGORICAL
    assert prod.metadata["feature_schema"]["names"] == CANONICAL_FEATURES  # registered metadata
    assert prod.metadata["feature_columns"] == CANONICAL_FEATURES
    ct = _base_pipeline(prod.model).named_steps["pre"]  # the artifact's own schema
    assert list(ct.feature_names_in_) == CANONICAL_FEATURES
    assert list(ct.transformers_[0][2]) == CANONICAL_NUMERIC
    assert list(ct.transformers_[1][2]) == CANONICAL_CATEGORICAL


def test_categorical_vocabulary_the_live_side_emits_matches_the_trained_encoder(prod):
    ct = _base_pipeline(prod.model).named_steps["pre"]
    enc = ct.transformers_[1][1]
    vocab = dict(
        zip(CANONICAL_CATEGORICAL, [set(map(str, c)) for c in enc.categories_], strict=True)
    )
    assert vocab["payment_method"] == set(TRAINED_PAYMENT_METHODS)
    assert vocab["consent_status"] == set(TRAINED_CONSENT_STATUSES)
    assert vocab["domain"] == {r.value for r in RiskCategory}


def test_preprocessing_is_the_training_preprocessing(prod):
    """Scaler statistics in the artifact equal those recomputed from the TRAIN split only."""
    if not DATA.exists():
        pytest.skip("dataset not generated")
    df = pd.read_csv(DATA)
    tr = df[df["split"] == "train"]
    ct = _base_pipeline(prod.model).named_steps["pre"]
    scaler = ct.transformers_[0][1].named_steps["scale"]
    imputer = ct.transformers_[0][1].named_steps["impute"]
    assert imputer.strategy == "median"
    i = CANONICAL_NUMERIC.index("amount_at_risk")
    # the scaler was fitted after median imputation; amount_at_risk has no missing values
    assert scaler.mean_[i] == pytest.approx(tr["amount_at_risk"].mean(), rel=1e-6)


def test_inference_path_reproduces_the_training_path_on_real_training_rows(prod):
    if not DATA.exists():
        pytest.skip("dataset not generated")
    df = pd.read_csv(DATA)
    df["suppression_status"] = df["suppression_status"].astype(int)
    sample = df[df["split"] == "test"].head(25)
    direct = prod.model.predict_proba(sample[CANONICAL_FEATURES])[:, 1]
    via_inference = []
    for _, row in sample.iterrows():
        vec = {
            k: (None if pd.isna(row[k]) else (row[k].item() if hasattr(row[k], "item") else row[k]))
            for k in CANONICAL_FEATURES
        }
        r = prod.predict(vec)
        assert r["status"] == "SUCCESS"
        via_inference.append(r["probability"])
    np.testing.assert_allclose(via_inference, direct, rtol=0, atol=1e-12)


def _uid():
    return uuid.uuid4().hex[:8]


def _mk_case(db, cust, created, amount=100.0):
    cid = f"par_{_uid()}"
    db.add(
        CaseModel(
            case_id=cid,
            customer_id=cust,
            risk_category="FAILED_PAYMENT",
            amount_at_risk=amount,
            current_state="POLICY_EVALUATED",
            created_at=created,
        )
    )
    db.flush()
    db.add(
        EventModel(
            event_id=f"e_{_uid()}",
            case_id=cid,
            customer_id=cust,
            external_system="t",
            external_event_id=f"x_{_uid()}",
            risk_category="FAILED_PAYMENT",
            amount=amount,
            timestamp=created,
            raw_payload={},
        )
    )
    db.flush()
    return cid


def _reference_generator_history(outcomes):
    """Independent oracle of the TRAINING definition (generate_recovery_outcome): running
    counters updated only AFTER each obligation's own label is decided; -1.0 when no history."""
    rec = non = 0
    rows = []
    for y in outcomes:
        total = rec + non
        rows.append((rec, non, rec / total if total else -1.0))
        rec, non = (rec + 1, non) if y else (rec, non + 1)
    return rows


def test_rolling_history_equals_the_training_generators_definition_step_by_step():
    db = SessionLocal()
    try:
        cust = f"par_c_{_uid()}"
        outcomes = [1, 0, 0, 1, 1, 0]
        expected = _reference_generator_history(outcomes)
        got = []
        for i, y in enumerate(outcomes):
            created = T0 + timedelta(days=20 * i)
            cid = _mk_case(db, cust, created, amount=100.0)
            fv = derive_recovery_features(db, cid, created + timedelta(days=1))
            got.append(
                (
                    fv.values["number_of_previous_recoveries"],
                    fv.values["number_of_previous_failed_attempts"],
                    fv.values["prior_recovery_rate"],
                )
            )
            if y:
                db.add(
                    OutcomeEventModel(
                        outcome_event_id=f"o_{_uid()}",
                        external_system="t",
                        external_event_id=f"s_{_uid()}",
                        domain="FAILED_PAYMENT",
                        kind="SETTLEMENT",
                        outcome_type="settled",
                        customer_id=cust,
                        amount=100,
                        occurred_at=created + timedelta(days=2),
                        match_status="MATCHED",
                        case_id=cid,
                        applied_amount=100,
                        effect="APPLIED",
                        provenance="test",
                    )
                )
            else:
                db.add(
                    AuditModel(
                        id=_uid(),
                        case_id=cid,
                        from_state="VERIFYING",
                        to_state="CLOSED_NOT_RECOVERED",
                        evidence={},
                        timestamp=created + timedelta(days=2),
                    )
                )
            db.flush()
        assert got[0][2] == COLD_START_PRIOR_RECOVERY_RATE
        for (er, en, ep), (gr, gn, gp) in zip(expected, got, strict=True):
            assert (gr, gn) == (er, en)
            assert gp == pytest.approx(ep)
    finally:
        db.rollback()
        db.close()


def test_values_outside_the_trained_vocabulary_are_left_missing_not_encoded_as_unseen():
    db = SessionLocal()
    try:
        cust = f"par_v_{_uid()}"
        cid = _mk_case(db, cust, T0)
        ev = db.query(EventModel).filter(EventModel.case_id == cid).first()
        ev.raw_payload = {"Payment_Method": "PayPal"}
        db.add(
            ContactPermissionModel(
                id=_uid(),
                customer_id=cust,
                channel="payment",
                status="SUPPRESSED",
                source="t",
                effective_at=T0 - timedelta(days=1),
                recorded_at=T0 - timedelta(days=1),
            )
        )
        db.flush()
        fv = derive_recovery_features(db, cid, T0 + timedelta(days=1))
        assert (
            fv.values["payment_method"] is None
            and fv.provenance["payment_method"].status == "MISSING"
        )
        assert fv.values["consent_status"] is None  # SUPPRESSED is not a trained consent category
        assert fv.values["suppression_status"] == 1  # ...it is carried by suppression_status
        ev.raw_payload = {"payment_method": " UPI "}
        db.flush()
        assert (
            derive_recovery_features(db, cid, T0 + timedelta(days=1)).values["payment_method"]
            == "upi"
        )
    finally:
        db.rollback()
        db.close()


def test_numeric_features_are_numbers_or_none_and_categoricals_are_strings_or_none():
    db = SessionLocal()
    try:
        cid = _mk_case(db, f"par_t_{_uid()}", T0)
        vals = derive_recovery_features(db, cid, T0 + timedelta(days=2)).values
        assert list(vals) == CANONICAL_FEATURES  # order
        for k in CANONICAL_NUMERIC:
            assert vals[k] is None or (
                isinstance(vals[k], (int, float)) and not isinstance(vals[k], bool)
            )
        for k in CANONICAL_CATEGORICAL:
            assert vals[k] is None or isinstance(vals[k], str)
    finally:
        db.rollback()
        db.close()


def test_the_amount_vs_customer_average_approximation_can_never_be_silently_dropped():
    """The live definition (amount / rolling mean of earlier amounts) is NOT identical to the
    training one (amount / a fixed latent per-customer typical amount). It must stay declared
    in the API response and in the provenance note."""
    from api.main import CANONICAL_APPROXIMATIONS

    assert any(
        "amount_vs_customer_average" in a and "NOT mathematically identical" in a
        for a in CANONICAL_APPROXIMATIONS
    )
    db = SessionLocal()
    try:
        cust = f"par_a_{_uid()}"
        _mk_case(db, cust, T0, amount=200.0)
        cid = _mk_case(db, cust, T0 + timedelta(days=5), amount=300.0)
        fv = derive_recovery_features(db, cid, T0 + timedelta(days=6))
        assert fv.values["amount_vs_customer_average"] == pytest.approx(1.5)
        assert "rolling mean" in fv.provenance["amount_vs_customer_average"].note
    finally:
        db.rollback()
        db.close()


def test_registered_schema_json_is_stable_and_machine_readable(prod):
    meta_files = list(Path(DEFAULT_REGISTRY_DIR).glob(f"{prod.model_id}_metadata.json"))
    meta = json.loads(meta_files[0].read_text())
    assert (
        meta["feature_schema"]["schema_sha256"] and meta["artifact_sha256"] == prod.artifact_sha256
    )
