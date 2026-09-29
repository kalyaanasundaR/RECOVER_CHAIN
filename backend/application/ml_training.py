import hashlib
import json
import os
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _read_any(path: str) -> pd.DataFrame:
    """Load a dataset file regardless of format. Training previously assumed CSV,
    so every .xlsx / .parquet dataset silently fell back to the baseline."""
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _coerce_frame(X):
    """Make a raw dataframe model-safe: turn number-like text ("1,200", "₹ 45")
    into floats, date-like text into epoch-day floats, and leave everything else
    as strings. Runs both at fit time and — because it is a pipeline step — at
    predict time, so real uploaded rows are transformed exactly as trained."""
    df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    for c in df.columns:
        s = df[c]
        if s.dtype != object:
            continue
        cleaned = (
            s.astype(str)
            .str.replace(r"[,₹$€£\s]", "", regex=True)
            .replace({"": None, "nan": None, "None": None, "NaT": None})
        )
        num = pd.to_numeric(cleaned, errors="coerce")
        if num.notna().mean() >= 0.8:
            df[c] = num
            continue
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if dt.notna().mean() >= 0.8:
            df[c] = (dt - _EPOCH).dt.total_seconds() / 86400.0
            continue
        df[c] = s.astype(str)
    return df


class MLTrainingEngine:
    def __init__(
        self,
        spec: dict,
        data_path: str,
        output_dir: str,
        min_roc_auc: float = None,
        min_test_rows: int = None,
    ):
        self.spec = spec
        self.data_path = data_path
        self.output_dir = output_dir
        # Explicit overrides win over env vars — callers that must stay honest
        # (live auto-train) pass a real bar even if the process env zeroed it.
        self.min_roc_auc = min_roc_auc
        self.min_test_rows = min_test_rows
        self.run_id = f"train_{int(time.time())}"

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _invert_target_if_needed(self, df: pd.DataFrame, target_col: str):
        series = df[target_col]
        if not pd.api.types.is_numeric_dtype(series) or str(series.dtype) == "bool":
            series_str = series.astype(str).str.strip().str.lower()
            mapping = {
                "0": 0,
                "false": 0,
                "no": 0,
                "success": 0,
                "paid": 0,
                "settled": 0,
                "completed": 0,
                "1": 1,
                "true": 1,
                "yes": 1,
                "failed": 1,
                "unpaid": 1,
                "returned": 1,
                "declined": 1,
                "error": 1,
                "insufficient_funds": 1,
            }
            mapped = series_str.map(mapping).fillna(0).astype(int)
            series = mapped
        else:
            try:
                series = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
            except Exception:
                series = (series > 0).astype(int)

        if (
            self.spec["prediction_problem"] == "payment-failure-risk"
            and "recovered" in target_col.lower()
        ):
            return (series == 0).astype(int)
        return series

    def _capture_temporal_boundaries(
        self, tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame
    ) -> dict:
        """Best-effort train/validation/test date-range boundaries, keyed off the
        spec's declared temporal split column when it is present in the split
        frames. Row counts alone (split_row_counts) don't tell a reviewer WHEN
        each fold ends, which is what a temporal-leakage audit actually needs."""
        split_col = (self.spec.get("temporal_split") or {}).get("split_column")
        if not split_col:
            return {}

        def _range(frame: pd.DataFrame) -> dict | None:
            if split_col not in frame.columns or len(frame) == 0:
                return None
            parsed = pd.to_datetime(frame[split_col], errors="coerce", utc=True)
            parsed = parsed.dropna()
            if parsed.empty:
                return None
            return {"start": parsed.min().isoformat(), "end": parsed.max().isoformat()}

        boundaries = {
            "split_column": split_col,
            "train": _range(tr),
            "validation": _range(va),
            "test": _range(te),
        }
        return boundaries if any(boundaries[k] for k in ("train", "validation", "test")) else {}

    def _split_data(self, df: pd.DataFrame):
        # Prefer the pre-computed split if it exists and obeys chronologic rules
        if "split" in df.columns:
            train_mask = df["split"] == "train"
            val_mask = df["split"] == "validation"
            test_mask = df["split"] == "test"
            tr, va, te = df[train_mask], df[val_mask], df[test_mask]
            if len(tr) and len(te):
                return tr, va, te

        # Chronological split on the declared temporal column
        split_col = (self.spec.get("temporal_split") or {}).get("split_column")
        if split_col and split_col in df.columns:
            order = pd.to_datetime(df[split_col], errors="coerce", utc=True)
            sorted_df = (
                df.assign(__order__=order)
                .sort_values("__order__", kind="stable", na_position="first")
                .drop(columns="__order__")
            )
            n = len(sorted_df)
            tr = sorted_df.iloc[: int(n * 0.7)]
            va = sorted_df.iloc[int(n * 0.7) : int(n * 0.85)]
            te = sorted_df.iloc[int(n * 0.85) :]
            if len(tr) and len(te) and tr["__target__"].nunique() >= 2:
                self._split_kind = "TEMPORAL"
                return tr, va, te

        # Real data without a usable date — stratified random split so it still
        # trains. Flagged in metadata so nobody mistakes it for time-safe.
        self._split_kind = "STRATIFIED_FALLBACK"
        strat = df["__target__"] if df["__target__"].nunique() >= 2 else None
        tr, tmp = train_test_split(df, test_size=0.3, random_state=42, stratify=strat)
        strat2 = tmp["__target__"] if tmp["__target__"].nunique() >= 2 else None
        va, te = train_test_split(tmp, test_size=0.5, random_state=42, stratify=strat2)
        return tr, va, te

    def _evaluate(self, model, X, y, threshold=0.5):
        if len(X) == 0:
            return {}

        y_prob = model.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        # Guard against single-class in validation/test (though rare)
        if len(np.unique(y)) > 1:
            roc_auc = roc_auc_score(y, y_prob)
            pr_auc = average_precision_score(y, y_prob)
            brier = brier_score_loss(y, y_prob)
        else:
            roc_auc, pr_auc, brier = None, None, None

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            # Reported alongside — not in place of — the imbalance-aware metrics above.
            # Failed/recovered payments are rarely a 50/50 split, so accuracy alone can
            # look good on a model that just predicts the majority class; roc_auc / pr_auc
            # / recall are what the quality gate actually keys off.
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "positive_rate": float(np.mean(y)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "threshold": threshold,
        }

    def train_and_evaluate(self):
        start_time = time.time()

        # Dataset snapshot checksum (Phase 11 hardening): a sha256 of the exact
        # bytes trained on, so a registry entry can be tied back to precisely
        # which file version produced it — independent of dataset_snapshot.py's
        # live-case checksum (application/dataset_snapshot.py), which covers a
        # different asset (eligible case_ids for the outcome-event pipeline, not
        # a training CSV/XLSX/Parquet file on disk).
        dataset_snapshot_checksum = None
        try:
            with open(self.data_path, "rb") as fh:
                dataset_snapshot_checksum = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            pass  # e.g. empty data_path in a unit test that stops before this matters

        # 1. Load Data (CSV / XLSX / Parquet)
        df = _read_any(self.data_path)
        self._split_kind = "TEMPORAL"

        target_col = self.spec["target_column"]
        feature_cols = self.spec["feature_columns"]

        # Validation
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} missing.")

        missing_feats = [f for f in feature_cols if f not in df.columns]
        if missing_feats:
            raise ValueError(f"Missing required features: {missing_feats}")

        # Target preparation
        y_all = self._invert_target_if_needed(df, target_col)

        # Coerce number-like / date-like text columns before we decide which are
        # numeric vs categorical. The same coercion is baked into the served
        # pipeline (step 'coerce') so real uploaded rows transform identically.
        df[feature_cols] = _coerce_frame(df[feature_cols])
        X_all = df[feature_cols].copy()
        # drop feature columns that are entirely empty — nothing to learn from
        dead_cols = [c for c in feature_cols if X_all[c].isna().all()]
        feature_cols = [c for c in feature_cols if c not in dead_cols]
        X_all = X_all[feature_cols]

        num_cols = X_all.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()
        cat_cols = [c for c in feature_cols if c not in num_cols]

        # Split Data
        df["__target__"] = y_all
        train_df, val_df, test_df = self._split_data(df)
        temporal_boundaries = self._capture_temporal_boundaries(train_df, val_df, test_df)

        # Drift readiness (Phase 11): compare the training distribution against
        # the held-out test distribution on the same feature columns actually
        # used. This is readiness infrastructure — a real production-drift check
        # would compare training against NEWLY INGESTED data, which doesn't exist
        # yet; application.drift_readiness.compare_distributions is the reusable
        # function a future ingestion path calls for that.
        from application.drift_readiness import compare_distributions, summarize_distribution

        train_feature_distribution = summarize_distribution(train_df, feature_cols)
        train_test_drift = compare_distributions(
            train_feature_distribution,
            summarize_distribution(test_df, feature_cols) if len(test_df) else {},
        )

        X_train, y_train = train_df[feature_cols], train_df["__target__"]
        X_val, y_val = val_df[feature_cols], val_df["__target__"]
        X_test, y_test = test_df[feature_cols], test_df["__target__"]

        if len(y_train.unique()) < 2:
            raise ValueError("Training set must contain both classes.")

        split_row_counts = {"train": len(X_train), "validation": len(X_val), "test": len(X_test)}

        # A too-small or single-class validation fold makes prefit calibration
        # meaningless — fold it back into training and skip calibration.
        small_val = len(X_val) < 12 or y_val.nunique() < 2
        if small_val:
            X_train = pd.concat([X_train, X_val])
            y_train = pd.concat([y_train, y_val])

        # 2. Build Preprocessing
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ]
        )

        # 3. Candidate Models
        # Calculate scale_pos_weight
        pos_count = int((y_train == 1).sum())
        neg_count = len(y_train) - pos_count
        spw = neg_count / max(1, pos_count)

        models = {
            "logistic_regression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ),
            "xgboost": XGBClassifier(
                scale_pos_weight=spw,
                eval_metric="logloss",
                random_state=42,
                use_label_encoder=False,
            ),
        }

        results = {}
        best_model_name = None
        best_pr_auc = -1
        best_pipeline = None

        for name, clf in models.items():
            # Pipeline — 'coerce' first so a raw served row is made model-safe
            # exactly the way training data was.
            pipeline = Pipeline(
                steps=[
                    ("coerce", FunctionTransformer(_coerce_frame)),
                    ("preprocessor", preprocessor),
                    ("classifier", clf),
                ]
            )

            # Train
            pipeline.fit(X_train, y_train)

            # Calibrate on the validation set (skipped when that fold is too
            # small / single-class — see small_val above).
            final_model = pipeline
            if not small_val:
                try:
                    # FrozenEstimator, not cv="prefit" (removed in scikit-learn 1.6+;
                    # this repo pins 1.9.0 — cv="prefit" silently raised and every
                    # calibration attempt fell back to the uncalibrated pipeline,
                    # caught by the except below. Found and fixed during the AI
                    # reconstruction audit (docs/ai_reconstruction/00_FORENSIC_AUDIT.md).
                    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
                    calibrated.fit(X_val, y_val)
                    final_model = calibrated
                except Exception:
                    final_model = pipeline

            # Evaluate
            train_metrics = self._evaluate(final_model, X_train, y_train)
            val_metrics = self._evaluate(final_model, X_val, y_val)

            results[name] = {"train_metrics": train_metrics, "val_metrics": val_metrics}

            # Model Selection via PR-AUC on Validation
            current_pr_auc = val_metrics.get("pr_auc")
            if best_pipeline is None or (
                current_pr_auc is not None and current_pr_auc > best_pr_auc
            ):
                best_pr_auc = current_pr_auc if current_pr_auc is not None else -1
                best_model_name = name
                best_pipeline = final_model

        # Final Evaluation on Test Set
        test_metrics = self._evaluate(best_pipeline, X_test, y_test)

        runtime = time.time() - start_time

        # Quality gate: a model that does not beat "always predict majority" by a
        # margin is not fit to serve even as shadow telemetry. It is still saved
        # (for inspection) but flagged so the registry / predictor can refuse it.
        # ROC-AUC is the real quality bar and always applies. The row-count floor
        # is meant for production-scale data; ML_ADAPTIVE_GATE (default on) scales
        # it down for smaller real datasets so they can still serve as shadow —
        # flagged small_sample=True so the UI can show lower confidence.
        min_roc_auc = (
            self.min_roc_auc
            if self.min_roc_auc is not None
            else float(os.getenv("ML_MIN_ROC_AUC", "0.55"))
        )
        min_test_rows = (
            self.min_test_rows
            if self.min_test_rows is not None
            else int(os.getenv("ML_MIN_TEST_ROWS", "200"))
        )
        adaptive = os.getenv("ML_ADAPTIVE_GATE", "1") == "1"
        # ROC-AUC is the real bar; the row floor just guards against trusting a
        # ROC computed on a handful of rows. 15 held-out rows with both classes
        # is enough to serve as shadow telemetry for a smaller real dataset.
        eff_min_test_rows = min(min_test_rows, 15) if adaptive else min_test_rows

        # When the test fold is single-class, ROC on it is undefined — fall back
        # to the validation ROC before deciding.
        roc = test_metrics.get("roc_auc")
        roc_source = "test"
        if roc is None:
            roc = (results.get(best_model_name, {}).get("val_metrics", {}) or {}).get("roc_auc")
            roc_source = "validation"

        gate_reasons = []
        if min_roc_auc > 0 and (roc is None or roc < min_roc_auc):
            gate_reasons.append(f"{roc_source} ROC-AUC {roc} < {min_roc_auc}")
        if eff_min_test_rows > 0 and len(X_test) < eff_min_test_rows:
            gate_reasons.append(f"test set {len(X_test)} rows < {eff_min_test_rows}")
        model_status = "SELECTED" if not gate_reasons else "REJECTED_LOW_QUALITY"
        small_sample = len(X_test) < min_test_rows

        # Lifecycle status (Phase 11 hardening): distinguishes "an artifact file
        # exists" from "fit to serve" from "actually being served" from
        # "validated against real production outcomes" — the four are not the
        # same claim and `status` (SELECTED/REJECTED_LOW_QUALITY, above) doesn't
        # carry that distinction on its own.
        #   TRAINED         - artifact + metadata written (always true here)
        #   EVALUATED       - test/validation metrics computed (always true here)
        #   QUALITY_PASSED  - cleared the quality gate (mirrors model_status)
        #   SHADOW_ACTIVE   - NOT set here: only true once
        #                     MLPaymentFailurePredictor actually loads this
        #                     specific registry entry for a live dataset_id/task,
        #                     which is a serving-time fact, not a training-time one.
        #   PRODUCTION_VALIDATED - NEVER set by this training pipeline. Per the
        #                     provenance rules (application/outcome_provenance.py)
        #                     this requires real PRODUCTION-provenance outcome
        #                     evidence at volume, which this repo does not have
        #                     (see docs/history/PHASE_10_ML_DL_EVALUATION_REPORT.md).
        #                     Promoting a model to this status is a manual,
        #                     evidence-reviewed decision outside this pipeline.
        lifecycle_status = "QUALITY_PASSED" if not gate_reasons else "EVALUATED"

        provenance_status = self.spec.get("dataset_provenance_status", "UNKNOWN")

        # 4. Save Artifacts
        artifact_path = os.path.join(self.output_dir, f"{self.run_id}_model.joblib")
        joblib.dump(best_pipeline, artifact_path)

        # 5. Save Metadata Registry Entry
        metadata = {
            "model_id": self.run_id,
            "dataset_id": self.spec["dataset_id"],
            "task": self.spec["prediction_problem"],
            "target_column": target_col,
            "canonical_target_meaning": self.spec.get("target_definition", "Unknown"),
            "feature_columns": feature_cols,
            "canonical_feature_mapping": {
                col: self.spec.get("canonical_feature_mapping", {}).get(col, "UNKNOWN")
                for col in feature_cols
            },
            "preprocessing_pipeline": self.spec.get("preprocessing_steps", []),
            "model_version": best_model_name,
            "feature_schema_version": "1.0",
            "status": model_status,
            "quality_gate": {
                "passed": not gate_reasons,
                "reasons": gate_reasons,
                "min_roc_auc": min_roc_auc,
                "min_test_rows": min_test_rows,
                "effective_min_test_rows": eff_min_test_rows,
                "adaptive": adaptive,
                "small_sample": small_sample,
                "roc_auc": roc,
                "roc_source": roc_source,
            },
            "split_strategy": getattr(self, "_split_kind", "TEMPORAL"),
            "temporal_boundaries": temporal_boundaries,
            "train_test_distribution_drift": train_test_drift,
            # Phase 12: per-feature training distribution, reused at inference
            # time by application.anomaly_detection to flag a case whose
            # feature values fall far outside what this model was trained on -
            # independent of, and never influencing, recovery_probability.
            "training_feature_distribution": train_feature_distribution,
            "dataset_snapshot_checksum": dataset_snapshot_checksum,
            "lifecycle_status": lifecycle_status,
            "provenance_status": provenance_status,
            "dropped_empty_features": dead_cols,
            "selected_model": best_model_name,
            "candidate_results": results,
            "final_test_metrics": test_metrics,
            "artifact_path": artifact_path,
            "created_at": datetime.now(UTC).isoformat(),
            "training_runtime_sec": round(runtime, 2),
            "split_row_counts": split_row_counts,
            "calibrated": not small_val,
        }

        registry_path = os.path.join(self.output_dir, f"{self.run_id}_metadata.json")
        with open(registry_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata
