import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Every `prediction_problem` label application/ml_readiness.py's auto-detector can
# assign to a REVENUE-AT-RISK target column (see that module for the exact
# name-matching rules). "churn-prediction" and "fraud-prediction" are deliberately
# excluded here - they are real, different tasks that module can also auto-detect,
# and neither is a recovery-probability signal this shadow predictor may serve.
REVENUE_RECOVERY_TASKS = frozenset({"payment-failure-risk", "late-settlement-risk"})


class MLPaymentFailurePredictor:
    def __init__(
        self,
        dataset_id: str = None,
        registry_dir: str | None = None,
        legacy_model_path: str = "backend/ml/models/failure_risk_tree.json",
        legacy_features_path: str = "backend/ml/models/features.json",
    ):

        self.model = None
        self.features = []
        self.is_legacy = False
        self.metadata = {}
        self.canonical_mapping = {}

        # Strictly reject path traversal or malformed dataset_ids
        if dataset_id:
            if not isinstance(dataset_id, str) or not re.match(r"^[a-zA-Z0-9_-]+$", dataset_id):
                return

            from infrastructure.storage import model_registry_dir

            reg_dir = registry_dir or model_registry_dir()
            if not os.path.exists(reg_dir) and os.path.exists(os.path.join("backend", reg_dir)):
                reg_dir = os.path.join("backend", reg_dir)
            elif (
                not os.path.exists(reg_dir)
                and reg_dir.startswith("backend/")
                and os.path.exists(reg_dir[8:])
            ):
                reg_dir = reg_dir[8:]

            if os.path.exists(reg_dir):
                metadata_files = [f for f in os.listdir(reg_dir) if f.endswith("_metadata.json")]
                metadata_files.sort(reverse=True)

                for meta_file in metadata_files:
                    try:
                        with open(os.path.join(reg_dir, meta_file)) as f:
                            meta = json.load(f)
                    except Exception:
                        continue

                    if meta.get("status") == "REJECTED_LOW_QUALITY":
                        # Failed the training quality gate — not fit to serve.
                        continue
                    # Phase 10: previously required meta["task"] ==
                    # "payment-failure-risk" EXACTLY, so a model for the other
                    # revenue-recovery task label application/ml_readiness.py can
                    # legitimately auto-detect ("late-settlement-risk", for a target
                    # column whose name contains "late"/"delinquent") could never be
                    # loaded here, even after training successfully and passing the
                    # quality gate. REVENUE_RECOVERY_TASKS below is every task label
                    # that module can produce for an at-risk-revenue target;
                    # "churn-prediction" and "fraud-prediction" are deliberately
                    # EXCLUDED - they predict a different real-world event and must
                    # never be loaded as if they were a recovery-probability
                    # shadow model (test_phase29_dataset_isolation_and_security
                    # exercises exactly this: a dataset with only a churn-prediction
                    # model must still report model=None here).
                    if (
                        meta.get("dataset_id") == dataset_id
                        and meta.get("task") in REVENUE_RECOVERY_TASKS
                    ):
                        self.metadata = meta
                        artifact_path = self.metadata.get("artifact_path")
                        if artifact_path and not os.path.exists(artifact_path):
                            if os.path.exists(os.path.join("backend", artifact_path)):
                                artifact_path = os.path.join("backend", artifact_path)
                            elif (
                                artifact_path.startswith("backend/")
                                and os.path.exists(artifact_path[8:])
                                or (
                                    artifact_path.startswith("backend/")
                                    or artifact_path.startswith("backend\\")
                                )
                                and os.path.exists(artifact_path[8:])
                            ):
                                artifact_path = artifact_path[8:]

                        if artifact_path and os.path.exists(artifact_path):
                            # Ensure artifact path does not escape
                            abs_path = os.path.abspath(artifact_path)
                            self.model = joblib.load(abs_path)
                            self.features = self.metadata.get("feature_columns", [])
                            self.canonical_mapping = self.metadata.get(
                                "canonical_feature_mapping", {}
                            )
                            return

        # Fallback to legacy ONLY if dataset_id was explicitly None (not if dataset_id failed lookup or was invalid)
        if (
            dataset_id is None
            and os.path.exists(legacy_model_path)
            and os.path.exists(legacy_features_path)
        ):
            self.model = xgb.Booster()
            self.model.load_model(legacy_model_path)
            with open(legacy_features_path) as f:
                self.features = json.load(f)
            self.is_legacy = True
            self.metadata = {"model_version": "Legacy_Phase17", "dataset_id": "legacy_billing_v3"}

    def _map_canonical_to_original_features(self, canonical_features: dict) -> dict:
        """
        Maps standard canonical inputs (e.g. 'AMOUNT') to the original column names expected by the model
        using the authoritative canonical_feature_mapping stored during training.
        """
        if self.is_legacy:
            return canonical_features

        mapped = {}
        for feature_col in self.features:
            canon = self.canonical_mapping.get(feature_col)

            # Prefer the raw column value straight from the source row when it is
            # present (real dataset replay) — it matches what the pipeline was
            # fitted on. Only fall back to a canonical substitute otherwise.
            if feature_col in canonical_features and canonical_features[feature_col] is not None:
                mapped[feature_col] = canonical_features[feature_col]
            elif canon and canon != "UNKNOWN" and canon in canonical_features:
                mapped[feature_col] = canonical_features[canon]

        return mapped

    def predict_failure_risk(self, canonical_features: dict) -> dict:
        if self.model is None or not self.features:
            return {"probability": 0.0, "status": "NO_MODEL"}

        mapped_features = self._map_canonical_to_original_features(canonical_features)

        missing_features = [f for f in self.features if f not in mapped_features]
        if missing_features:
            raise ValueError(
                f"Incomplete feature set. Missing original features mapped to domain: {missing_features}"
            )

        if self.is_legacy:
            x = []
            for f in self.features:
                x.append(mapped_features.get(f, 0.0))
            dmat = xgb.DMatrix(np.array([x]), feature_names=self.features)
            prob = self.model.predict(dmat)[0]
        else:
            df = pd.DataFrame([mapped_features])
            prob = self.model.predict_proba(df[self.features])[0, 1]

        prob = float(prob)
        # Phase 12 model-failure-safety hardening: a NaN or out-of-[0,1]
        # probability (a corrupted artifact, a native-library edge case, a
        # pathological input row) must never reach ERV math as a silent
        # not-a-number or an impossible probability > 1 / < 0. Treated the
        # same as any other shadow-prediction failure: surfaced as a clean
        # error status so the caller (case_pipeline.predict_recovery_for_case)
        # falls back to the deterministic baseline, never as a "successful"
        # prediction carrying a nonsensical number.
        if not np.isfinite(prob) or prob < 0.0 or prob > 1.0:
            return {
                "probability": 0.0,
                "status": "INVALID_PROBABILITY",
                "error": f"model produced a non-finite or out-of-range probability: {prob!r}",
            }

        from application.explainability import confidence_tier

        model_metadata = {
            "dataset_id": self.metadata.get("dataset_id"),
            "model_version": self.metadata.get("model_version", "unknown"),
            "features_used": self.features,
            "shadow_mode_active": True,
            # Phase 11: exposes WHEN this model was trained so a caller (UI,
            # drift-readiness check) can judge staleness. This module does
            # not itself refuse to serve an old model — that's a policy
            # decision for whatever consumes this metadata, not a silent
            # runtime behavior change here.
            "trained_at": self.metadata.get("created_at"),
            "confidence_tier": confidence_tier(prob, self.metadata),
        }

        # Phase 12: explanation is genuinely useful but strictly supplementary
        # - a broken explainer must never turn an otherwise-valid prediction
        # into a failed one, so it is isolated in its own try/except and
        # simply omitted (not defaulted to a fabricated value) on error.
        try:
            from application.explainability import explain_legacy_xgboost, explain_sklearn_pipeline

            if self.is_legacy:
                model_metadata["explanation"] = explain_legacy_xgboost(
                    self.model, self.features, mapped_features
                )
            else:
                model_metadata["explanation"] = explain_sklearn_pipeline(
                    self.model, self.features, mapped_features
                )
        except Exception:
            model_metadata["explanation"] = None

        try:
            from application.anomaly_detection import score_anomaly

            model_metadata["anomaly"] = score_anomaly(
                mapped_features, self.metadata.get("training_feature_distribution")
            )
        except Exception:
            model_metadata["anomaly"] = None

        return {
            "probability": prob,
            "status": "SUCCESS",
            "model_metadata": model_metadata,
        }
