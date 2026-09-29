from datetime import UTC, datetime
from typing import Any

import pandas as pd

from application.dataset_intelligence import CanonicalField


class MLReadinessAnalyzer:
    @staticmethod
    def analyze_readiness(ds, file_path: str) -> dict[str, Any]:
        """
        Analyzes an already-profiled dataset to determine ML readiness and generate
        a rigorous training specification.
        """
        if not ds.columns_profile:
            raise ValueError("Dataset has not been profiled yet.")

        profile = ds.columns_profile
        schema = ds.recoverchain_signals or []
        leakage = ds.leakage_detection or []

        target_column = None
        prediction_problem = "unknown"
        target_definition = "Unknown outcome"

        # 1. Target Discovery
        # Look for explicitly mapped outcomes or common target names
        for col in schema:
            if col["canonical_field"] == CanonicalField.OUTCOME.value:
                target_column = col["original_column"]
                break

        if not target_column:
            # Fallback heuristic
            for c in profile:
                if "target" in c["column_name"].lower():
                    target_column = c["column_name"]
                    break

        if target_column:
            t_lower = target_column.lower()
            if "late" in t_lower or "delinquent" in t_lower:
                prediction_problem = "late-settlement-risk"
                target_definition = (
                    "Predicts risk of an invoice/payment settling late (but eventually settling)."
                )
            elif "churn" in t_lower:
                prediction_problem = "churn-prediction"
                target_definition = "Predicts loss of subscriber/customer."
            elif "fraud" in t_lower:
                prediction_problem = "fraud-prediction"
                target_definition = "Predicts malicious transaction behavior."
            else:
                prediction_problem = "payment-failure-risk"
                target_definition = "Predicts risk of payment failure before billing attempt."

        # 2. Feature Eligibility
        excluded_columns = []
        exclusion_reasons = {}
        feature_columns = []

        # We need a quick read for class imbalance (CSV / XLSX / Parquet)
        class_balance = {}
        if target_column:
            try:
                p = str(file_path).lower()
                if p.endswith(".parquet"):
                    tgt = pd.read_parquet(file_path, columns=[target_column])[target_column]
                elif p.endswith((".xlsx", ".xls")):
                    tgt = pd.read_excel(file_path, usecols=[target_column])[target_column]
                else:
                    tgt = pd.concat(
                        [
                            c[target_column]
                            for c in pd.read_csv(
                                file_path, usecols=[target_column], chunksize=100000
                            )
                        ]
                    )
                val_counts = tgt.value_counts()
                total = val_counts.sum()
                if total > 0:
                    pos = int(val_counts.max())
                    neg = int(val_counts.min()) if len(val_counts) > 1 else 0
                    minority_rate = (neg / total) * 100 if total > 0 else 0
                    class_balance = {
                        "positive_rate": round((pos / total) * 100, 2),
                        "negative_rate": round(minority_rate, 2),
                        "imbalance_ratio": f"{round(pos / max(1, neg), 1)}:1" if neg > 0 else "N/A",
                        "minority_count": neg,
                    }
            except Exception as e:
                class_balance = {"error": f"Failed to calculate class balance: {str(e)}"}

        # Classify columns
        leakage_map = {l["column"]: l["reason"] for l in leakage}

        entity_col = None
        time_col = None

        for c in profile:
            c_name = c["column_name"]

            # Identify entity and time
            c_schema = next((s for s in schema if s["original_column"] == c_name), None)
            if c_schema:
                if c_schema["canonical_field"] in [
                    CanonicalField.ENTITY_ID.value,
                    CanonicalField.ACCOUNT_ID.value,
                    CanonicalField.CUSTOMER_ID.value,
                ]:
                    entity_col = c_name
                elif c_schema["canonical_field"] in [
                    CanonicalField.TIMESTAMP.value,
                    CanonicalField.SETTLEMENT_DATE.value,
                ]:
                    time_col = c_name

            if c_name == target_column:
                excluded_columns.append(c_name)
                exclusion_reasons[c_name] = "TARGET"
                continue

            # Phase 10: a column that IS the train/validation/test partition label
            # (exactly what application.ml_training.MLTrainingEngine._split_data
            # itself reads and writes) must never be treated as a feature - a live
            # inference request has no such column at all, and using it as a
            # feature at training time only pollutes the model with a partition
            # indicator, never a real predictive signal.
            if c_name.strip().lower() == "split":
                excluded_columns.append(c_name)
                exclusion_reasons[c_name] = "DATA_PARTITION_LABEL"
                continue

            if c_name in leakage_map:
                excluded_columns.append(c_name)
                exclusion_reasons[c_name] = (
                    f"POST_OUTCOME / TEMPORALLY_AMBIGUOUS: {leakage_map[c_name]}"
                )
                continue

            if c.get("unique_count", 0) <= 1:
                excluded_columns.append(c_name)
                exclusion_reasons[c_name] = "CONSTANT"
                continue

            if c_schema and c_schema["canonical_field"] in [
                CanonicalField.ENTITY_ID.value,
                CanonicalField.ACCOUNT_ID.value,
                CanonicalField.CUSTOMER_ID.value,
            ]:
                excluded_columns.append(c_name)
                exclusion_reasons[c_name] = "IDENTIFIER"
                continue

            if c_schema and c_schema["canonical_field"] == CanonicalField.TRANSACTION_ID.value:
                excluded_columns.append(c_name)
                exclusion_reasons[c_name] = "IDENTIFIER"
                continue

            if c_schema and c_schema["canonical_field"] == CanonicalField.SETTLEMENT_DATE.value:
                excluded_columns.append(c_name)
                exclusion_reasons[c_name] = "POST_OUTCOME / SETTLEMENT_DATE"
                continue

            if c["dtype"] == "object" and c.get("unique_count", 0) > 1000:
                excluded_columns.append(c_name)
                exclusion_reasons[c_name] = "HIGH_CARDINALITY"
                continue

            feature_columns.append(c_name)

        # 3. Splitting Strategy
        if time_col and entity_col:
            temporal_split = {
                "strategy": "TEMPORAL_WITH_ENTITY_ISOLATION",
                "train_period": "earliest 70%",
                "validation_period": "middle 15%",
                "test_period": "latest 15%",
                "split_column": time_col,
            }
            entity_strategy = (
                "Group by ENTITY_ID to prevent train/test leakage of future entity states."
            )
        elif time_col:
            temporal_split = {
                "strategy": "TEMPORAL_CHRONOLOGICAL",
                "train_period": "earliest 70%",
                "validation_period": "middle 15%",
                "test_period": "latest 15%",
                "split_column": time_col,
            }
            entity_strategy = "No explicit entity grouping detected."
        else:
            temporal_split = {
                "strategy": "RANDOM_SHUFFLE",
                "train_period": "70%",
                "validation_period": "15%",
                "test_period": "15%",
                "warning": "Unsafe: No temporal column detected, random shuffle risks time leakage.",
            }
            entity_strategy = "Unknown"

        # 4. Metrics & Status
        warnings = []
        if not target_column:
            status = "NOT_SUPERVISED_LEARNING"
            metrics = []
            warnings.append("No valid target outcome detected.")
        else:
            is_imbalanced = class_balance.get("negative_rate", 50) < 10
            metrics = (
                ["ROC-AUC", "PR-AUC", "precision", "recall"]
                if is_imbalanced
                else ["accuracy", "F1"]
            )
            if is_imbalanced:
                warnings.append(
                    "Target is highly imbalanced. Avoid raw accuracy as primary metric."
                )

            if not feature_columns:
                status = "INSUFFICIENT_DATA"
                warnings.append(
                    "No safe predictive features remain after leakage and identifier exclusion."
                )
            elif not time_col:
                status = "LEAKAGE_RISK"
                warnings.append(
                    "No temporal column detected; historical cutoff enforcement impossible."
                )
            else:
                status = "ML_TRAINING_READY" if not warnings else "ML_TRAINING_READY_WITH_WARNINGS"

        preprocessing = [
            "Impute missing numeric values via grouped historical medians strictly before cutoff.",
            "Encode low-cardinality categoricals (OHE or Target Encoding securely inside CV folds).",
            "Generate rolling historical features strictly partitioned by entity and time cutoff.",
        ]

        canonical_mapping = {
            s["original_column"]: s["canonical_field"]
            for s in schema
            if s["original_column"] in feature_columns
        }

        return {
            "dataset_id": ds.dataset_id,
            "target_column": target_column,
            "target_definition": target_definition,
            "prediction_problem": prediction_problem,
            "feature_columns": feature_columns,
            "excluded_columns": excluded_columns,
            "exclusion_reasons": exclusion_reasons,
            "canonical_feature_mapping": canonical_mapping,
            "preprocessing_steps": preprocessing,
            "temporal_split": temporal_split,
            "entity_strategy": entity_strategy,
            "class_balance": class_balance,
            "recommended_metrics": metrics,
            "warnings": warnings,
            "readiness_status": status,
            "analysis_timestamp": datetime.now(UTC).isoformat(),
        }
