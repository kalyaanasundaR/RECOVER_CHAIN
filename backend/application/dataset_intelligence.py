import re
from enum import Enum
from typing import Any

import pandas as pd


class CanonicalField(str, Enum):
    CUSTOMER_ID = "CUSTOMER_ID"
    ACCOUNT_ID = "ACCOUNT_ID"
    ENTITY_ID = "ENTITY_ID"
    TRANSACTION_ID = "TRANSACTION_ID"
    TIMESTAMP = "TIMESTAMP"
    SETTLEMENT_DATE = "SETTLEMENT_DATE"
    AMOUNT = "AMOUNT"
    BALANCE = "BALANCE"
    CURRENCY = "CURRENCY"
    STATUS = "STATUS"
    PAYMENT_METHOD = "PAYMENT_METHOD"
    OUTCOME = "OUTCOME"
    TARGET = "TARGET"
    # A column naming each row's revenue domain (Phase 4). Only ever set by an explicit
    # mapping confirmation; see application/domain_classification.py.
    EVENT_CATEGORY = "EVENT_CATEGORY"
    UNKNOWN = "UNKNOWN"


class DatasetClassification(str, Enum):
    ANALYSIS_READY = "ANALYSIS_READY"
    ML_TRAINING_READY = "ML_TRAINING_READY"
    ML_TRAINING_READY_WITH_EXCLUSIONS = "ML_TRAINING_READY_WITH_EXCLUSIONS"
    PARTIALLY_USABLE = "PARTIALLY_USABLE"
    INSUFFICIENT = "INSUFFICIENT"


class SemanticMapper:
    def map_schema(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        mappings = []
        for col in df.columns:
            m = self.map_column(col, df[col])
            mappings.append(m)

        single_use = [
            CanonicalField.AMOUNT.value,
            CanonicalField.TARGET.value,
            CanonicalField.OUTCOME.value,
            CanonicalField.TIMESTAMP.value,
            CanonicalField.SETTLEMENT_DATE.value,
        ]

        canonical_to_cols = {}
        for idx, m in enumerate(mappings):
            cf = m["canonical_field"]
            if cf not in canonical_to_cols:
                canonical_to_cols[cf] = []
            canonical_to_cols[cf].append((idx, m["confidence"], m["original_column"]))

        for cf in single_use:
            if cf in canonical_to_cols and len(canonical_to_cols[cf]) > 1:
                conf_weights = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
                col_entries = canonical_to_cols[cf]
                max_w = max(conf_weights.get(c[1], 0) for c in col_entries)
                top_entries = [c for c in col_entries if conf_weights.get(c[1], 0) == max_w]

                if len(top_entries) == 1:
                    top_idx = top_entries[0][0]
                    for idx, _conf, _col_name in col_entries:
                        if idx != top_idx:
                            mappings[idx]["canonical_field"] = CanonicalField.UNKNOWN.value
                            mappings[idx]["reason"] = (
                                "Downgraded due to conflict with a higher confidence column."
                            )
                else:
                    for idx, _conf, _col_name in col_entries:
                        mappings[idx]["canonical_field"] = CanonicalField.UNKNOWN.value
                        mappings[idx]["confidence"] = "LOW"
                        mappings[idx]["reason"] = (
                            "AMBIGUOUS_MAPPING: Multiple columns plausibly map to this field. Requires user confirmation."
                        )

        return mappings

    def map_column(self, col_name: str, series: pd.Series) -> dict[str, Any]:
        # 1. Normalize column name
        camel_spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(col_name)).lower()
        norm_col = camel_spaced.replace(" ", "_").replace("-", "_")
        tokens = set(re.split(r"[^a-zA-Z0-9]+", camel_spaced))
        tokens.discard("")

        dtype = str(series.dtype)
        is_numeric = "int" in dtype or "float" in dtype
        if not is_numeric and ("object" in dtype or "str" in dtype) and not series.empty:
            sample_clean = (
                series.dropna()
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            coerced = pd.to_numeric(sample_clean, errors="coerce")
            if coerced.notnull().sum() >= max(1, int(len(sample_clean) * 0.5)):
                is_numeric = True
        is_datetime = "datetime" in dtype
        is_bool = "bool" in dtype

        total_cnt = len(series)
        unique_cnt = int(series.nunique())
        unique_rate = unique_cnt / max(1, total_cnt)

        candidates = {}

        # Check if column is clearly a post-outcome / leakage column
        leakage_keywords = [
            "actual_recovered",
            "recovered_amount",
            "actual_amount_recovered",
            "actual_recovered_amount",
            "recovery_date",
            "post_recovery",
            "settled_amount",
            "final_settlement",
            "chargeback_date",
            "final_status",
            "collection_result",
        ]
        is_leakage_col = any(k in norm_col for k in leakage_keywords)

        # ----------------------------------------------------
        # 1. TIMESTAMP / SETTLEMENT_DATE
        # ----------------------------------------------------
        settle_words = {"settlement", "settled", "cleared"}
        time_words = {
            "timestamp",
            "datetime",
            "date",
            "time",
            "created",
            "event",
            "txn_time",
            "paid",
            "processed",
            "txn_datetime",
        }
        has_settle = any(w in norm_col for w in settle_words)
        has_time = any(w in tokens for w in time_words) or any(
            w in norm_col
            for w in [
                "_date",
                "date_",
                "_time",
                "time_",
                "_at",
                "created_at",
                "attempted_at",
                "attempted",
                "initiated",
                "occurred",
                "paid_on",
                "processed_on",
                "event_time",
                "txn_time",
                "transaction_date",
                "invoice_date",
                "event_date",
                "txn_datetime",
            ]
        )

        time_score = 0.0
        evidence_time = []

        is_date_values = False
        if is_datetime:
            is_date_values = True
            time_score += 0.5
            evidence_time.append("Datatype is explicitly datetime.")
        elif dtype == "object" and not series.empty:
            sample = series.dropna().head(10).astype(str)
            date_pattern = r"^\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{4}/\d{2}/\d{2}"
            matches = sum(1 for s in sample if re.search(date_pattern, s))
            if matches >= max(1, len(sample) // 2):
                is_date_values = True
                time_score += 0.4
                evidence_time.append("Values match standard date patterns.")

        if has_time or is_date_values:
            if has_time:
                time_score += 0.5
                evidence_time.append("Name implies temporal/date concept.")
            if has_settle:
                candidates[CanonicalField.SETTLEMENT_DATE.value] = (
                    time_score + 0.1,
                    evidence_time + ["Specific settlement time semantics."],
                )
            else:
                candidates[CanonicalField.TIMESTAMP.value] = (
                    time_score if not is_leakage_col else 0.35,
                    evidence_time
                    if not is_leakage_col
                    else evidence_time + ["Post-outcome temporal attribute."],
                )

        # ----------------------------------------------------
        # 2. AMOUNT / BALANCE / CURRENCY
        # ----------------------------------------------------
        amount_words = {
            "amount",
            "amt",
            "price",
            "value",
            "total",
            "fee",
            "debit",
            "credit",
            "invoice_total",
            "txn_value",
            "payment_value",
            "monetary_value",
            "payment_amt",
            "debit_value",
            "bill_value",
            "invoice_amount",
        }
        balance_words = {
            "balance",
            "bal",
            "outstanding_balance",
            "avail_bal",
            "current_balance",
            "post_recovery_balance",
        }
        currency_words = {"currency", "curr", "ccy"}
        count_words = {
            "count",
            "num",
            "number_of",
            "attempts",
            "index",
            "qty",
            "quantity",
            "rate",
            "ratio",
            "pct",
            "percent",
            "score",
            "days",
            "age",
        }

        has_count = any(w in tokens for w in count_words) or any(
            w in norm_col for w in ["_count", "count_", "num_", "_qty"]
        )
        has_amt_name = any(w in tokens for w in amount_words) or any(
            w in norm_col
            for w in [
                "_amount",
                "amount_",
                "_amt",
                "amt_",
                "_value",
                "value_",
                "invoice_total",
                "txn_value",
                "payment_value",
                "monetary_value",
                "payment_amt",
                "debit_value",
                "bill_value",
                "invoice_amount",
                "price",
                "fee",
                "total",
            ]
        )
        has_bal_name = any(w in tokens for w in balance_words) or "balance" in norm_col
        has_curr_name = any(w in tokens for w in currency_words) or "currency" in norm_col

        if has_curr_name or (
            dtype == "object"
            and unique_cnt <= 10
            and series.dropna().head(5).astype(str).str.len().max() == 3
        ):
            candidates[CanonicalField.CURRENCY.value] = (
                0.8,
                ["Name or values match currency codes."],
            )

        if not has_count and not is_datetime:
            if has_bal_name:
                bal_score = 0.5
                ev_bal = ["Name implies account balance."]
                if is_numeric:
                    bal_score += 0.4
                    ev_bal.append("Datatype is numeric.")
                candidates[CanonicalField.BALANCE.value] = (bal_score, ev_bal)

            if has_amt_name:
                ev_amt = ["Name implies monetary transaction amount."]
                if is_numeric:
                    amt_score = 0.9
                    ev_amt.append("Datatype is numeric.")
                else:
                    amt_score = 0.35
                    ev_amt.append("Datatype is non-numeric string.")
                if is_leakage_col:
                    amt_score = 0.35
                    ev_amt.append("Leaked post-recovery attribute.")
                candidates[CanonicalField.AMOUNT.value] = (amt_score, ev_amt)
            elif is_numeric and "float" in dtype and unique_cnt > 10 and not is_date_values:
                candidates[CanonicalField.AMOUNT.value] = (
                    0.35,
                    ["Float distribution matches continuous amounts."],
                )

        # ----------------------------------------------------
        # 3. IDENTIFIERS (CUSTOMER_ID, ACCOUNT_ID, ENTITY_ID, TRANSACTION_ID)
        # ----------------------------------------------------
        cust_words = {
            "customer",
            "client",
            "cust",
            "user",
            "member",
            "party",
            "debtor",
            "subscriber",
            "seller",
            "shopper",
            "buyer",
            "payer",
            "payee",
            "merchant",
            "vendor",
            "borrower",
            "holder",
            "tenant",
            "contact",
            "company",
            "org",
            "account_holder",
        }
        acct_words = {"account", "acc", "acct"}
        entity_words = {"entity"}
        tx_words = {"transaction", "txn", "tx", "payment", "receipt", "order", "event", "invoice"}
        id_words = {"id", "ref", "key", "no", "num", "number", "code", "identifier", "reference"}

        has_cust = any(w in tokens for w in cust_words) or any(
            w in norm_col for w in ["customer", "client", "cust_", "_cust", "user_"]
        )
        has_acct = any(w in tokens for w in acct_words) or any(
            w in norm_col for w in ["account", "acct", "acc_", "_acc", "_acct", "acc_no", "acct_no"]
        )
        has_entity = any(w in tokens for w in entity_words) or "entity" in norm_col
        has_tx = any(w in tokens for w in tx_words) or any(
            w in norm_col
            for w in [
                "transaction",
                "txn_",
                "_txn",
                "tx_",
                "_tx",
                "payment_id",
                "receipt_no",
                "order_id",
                "invoice_id",
            ]
        )
        has_id = any(w in tokens for w in id_words) or any(
            w in norm_col
            for w in [
                "_id",
                "id_",
                "_ref",
                "ref_",
                "_key",
                "_no",
                "no_",
                "_num",
                "number",
                "identifier",
                "reference",
            ]
        )

        if not is_datetime and not has_amt_name and not has_bal_name:
            if has_cust:
                c_score = 0.6 if has_id else 0.4
                ev_c = ["Name implies customer/client entity."]
                if unique_rate > 0.3 or not is_numeric:
                    c_score += 0.3
                    ev_c.append("High uniqueness confirms identifier behavior.")
                candidates[CanonicalField.CUSTOMER_ID.value] = (c_score, ev_c)

            elif has_acct:
                a_score = 0.6 if has_id else 0.4
                ev_a = ["Name implies account entity."]
                if unique_rate > 0.3 or not is_numeric:
                    a_score += 0.3
                    ev_a.append("High uniqueness confirms identifier behavior.")
                candidates[CanonicalField.ACCOUNT_ID.value] = (a_score, ev_a)

            elif has_entity:
                candidates[CanonicalField.ENTITY_ID.value] = (
                    0.8,
                    ["Name implies generic entity identifier."],
                )

            elif has_tx and has_id:
                candidates[CanonicalField.TRANSACTION_ID.value] = (
                    0.9,
                    ["Name explicitly implies transaction identifier."],
                )

            elif has_id and not has_time and not is_date_values:
                if unique_rate >= 0.95:
                    candidates[CanonicalField.TRANSACTION_ID.value] = (
                        0.7,
                        ["Generic identifier with ~100% uniqueness suggests Transaction ID."],
                    )
                elif unique_rate >= 0.4:
                    candidates[CanonicalField.ACCOUNT_ID.value] = (
                        0.5,
                        ["Generic identifier suggests Entity/Account ID."],
                    )

        # ----------------------------------------------------
        # 4. PAYMENT_METHOD / STATUS / OUTCOME / TARGET
        # ----------------------------------------------------
        # 4a. Value-based OUTCOME detection — a low-cardinality column whose
        # values look like pass/fail states is an outcome no matter what it is
        # named (paid_status, settlement, result, is_paid, state, ...).
        # Unambiguous outcome words — name does not matter if the values say this.
        OUTCOME_STRONG = {
            "paid",
            "unpaid",
            "failed",
            "fail",
            "failure",
            "success",
            "successful",
            "returned",
            "declined",
            "decline",
            "settled",
            "completed",
            "pending",
            "overdue",
            "collected",
            "chargeback",
            "refunded",
            "recovered",
            "not_recovered",
            "won",
            "lost",
            "delinquent",
            "default",
        }
        # Ambiguous booleans — only an outcome if the column name also hints at it.
        OUTCOME_WEAK = {
            "yes",
            "no",
            "true",
            "false",
            "0",
            "1",
            "y",
            "n",
            "t",
            "f",
            "paid",
            "unpaid",
        }
        val_set = set()
        if (
            not is_datetime
            and not is_date_values
            and (
                (not is_numeric and 2 <= unique_cnt <= 6)
                or is_bool
                or (is_numeric and unique_cnt == 2)
            )
        ):
            val_set = set(series.dropna().astype(str).str.strip().str.lower().unique())
        name_hints_outcome = (
            any(
                t
                in (
                    "status",
                    "state",
                    "result",
                    "outcome",
                    "paid",
                    "failed",
                    "flag",
                    "settlement",
                    "settled",
                    "target",
                    "label",
                )
                for t in tokens
            )
            or "paid" in norm_col
            or "status" in norm_col
        )
        strong_hit = bool(val_set) and (len(val_set & OUTCOME_STRONG) / len(val_set)) >= 0.5
        weak_hit = (
            bool(val_set)
            and name_hints_outcome
            and (len(val_set & OUTCOME_WEAK) / len(val_set)) >= 0.6
        )
        if strong_hit or weak_hit:
            candidates[CanonicalField.OUTCOME.value] = (
                0.92 if not is_leakage_col else 0.4,
                [f"Values look like pass/fail outcomes: {', '.join(sorted(val_set)[:6])}."],
            )

        pay_method_words = {"payment_method", "payment_type", "pay_method", "card_type", "pay_type"}
        has_pay_method = any(w in tokens for w in pay_method_words) or any(
            w in norm_col for w in ["payment_method", "payment_type", "pay_method", "card_type"]
        )
        if has_pay_method and unique_cnt <= 20:
            candidates[CanonicalField.PAYMENT_METHOD.value] = (
                0.85,
                ["Name implies payment method channel."],
            )

        target_words = {"target", "label", "target_late"}
        outcome_words = {
            "outcome",
            "failed",
            "is_failed",
            "payment_failed",
            "failure",
            "failure_indicator",
            "recovery_flag",
            "default_flag",
            "settlement_result",
            "payment_result",
            "recovery_status",
            "result",
            "delinquent",
            "churn",
            "is_churn",
            "payment_status",
            "paid_status",
            "pay_status",
            "paid",
            "unpaid",
            "is_paid",
            "settlement",
            "settled",
            "collected",
            "late",
            "target_late",
        }
        status_words = {"status", "state", "stage", "current_status", "lifecycle_stage"}

        has_target = any(w in tokens for w in target_words) or any(
            w in norm_col for w in ["target", "label", "target_late"]
        )
        has_outcome = any(w in tokens for w in outcome_words) or any(
            w in norm_col
            for w in [
                "outcome",
                "failed",
                "failure",
                "settlement_result",
                "payment_result",
                "recovery_status",
                "payment_status",
                "paid_status",
                "pay_status",
                "is_paid",
                "is_failed",
                "recovery_flag",
                "target_late",
            ]
        )
        has_status = any(w in tokens for w in status_words) or "status" in norm_col

        # A name-only OUTCOME hit (no supporting values) must still look like a
        # label: not a date, and low cardinality. This stops "failure" inside
        # "prior_failures" / "failure_code" and "settlement" inside
        # "settlement_date" from being mistaken for the target.
        if not (strong_hit or weak_hit):
            if is_datetime or is_date_values or unique_cnt > 3:
                has_target = has_outcome = False

        is_binary_or_low_card = is_bool or (
            2 <= unique_cnt <= 10 and not is_datetime and not is_date_values
        )

        if (
            (has_target or has_outcome or has_status)
            and not has_pay_method
            and not is_datetime
            and not is_date_values
        ):
            if unique_cnt > 15:
                if has_status:
                    candidates[CanonicalField.STATUS.value] = (
                        0.7,
                        ["High-cardinality status/state field."],
                    )
                else:
                    candidates[CanonicalField.UNKNOWN.value] = (
                        0.1,
                        [
                            "Outcome/target column has too many unique values for binary classification."
                        ],
                    )
            elif has_outcome or has_target or (has_status and is_binary_or_low_card):
                o_score = 0.6
                ev_o = ["Name implies financial recovery/payment outcome or target."]
                if is_binary_or_low_card:
                    o_score += 0.3
                    ev_o.append(
                        f"Low cardinality ({unique_cnt}) confirms categorical outcome/target."
                    )
                candidates[CanonicalField.OUTCOME.value] = (
                    o_score if not is_leakage_col else 0.35,
                    ev_o if not is_leakage_col else ev_o + ["Post-outcome status attribute."],
                )
            elif has_status:
                candidates[CanonicalField.STATUS.value] = (
                    0.6,
                    ["Name implies operational status."],
                )

        # Pick best match
        best_field = CanonicalField.UNKNOWN.value
        best_score = 0.0
        best_evidence = []

        for field, (score, ev) in candidates.items():
            if score > best_score:
                best_score = score
                best_field = field
                best_evidence = ev

        confidence = "UNKNOWN"
        reason = "Insufficient evidence to determine semantic meaning."

        if best_score >= 0.75:
            confidence = "HIGH"
            reason = "Strong semantic and statistical evidence."
        elif best_score >= 0.4:
            confidence = "MEDIUM"
            reason = "Moderate evidence or conflicting signals."
        elif best_score > 0:
            confidence = "LOW"
            reason = "Weak evidence, requires manual confirmation."

        if best_score < 0.3 or best_field == CanonicalField.UNKNOWN.value:
            best_field = CanonicalField.UNKNOWN.value
            confidence = "LOW" if best_score > 0 else "UNKNOWN"
            reason = "Insufficient evidence to determine semantic meaning."
            best_evidence = []

        return {
            "original_column": col_name,
            "canonical_field": best_field,
            "confidence": confidence,
            "reason": reason,
            "evidence": best_evidence,
        }


class DatasetValidator:
    @staticmethod
    def detect_leakage(col_name: str, canonical_field: str) -> dict[str, str]:
        camel_spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(col_name)).lower()
        norm_col = camel_spaced.replace(" ", "_").replace("-", "_")

        if canonical_field == CanonicalField.SETTLEMENT_DATE.value:
            return {
                "status": "WARNING",
                "reason": "POST_OUTCOME: Settlement date occurs after the failure event.",
                "severity": "HIGH",
            }

        leakage_patterns = [
            "recovery",
            "recovered",
            "actual_recovered",
            "actual_amount_recovered",
            "actual_recovered_amount",
            "recovery_date",
            "final_status",
            "settled_amount",
            "post_recovery_balance",
            "final_settlement_status",
            "collection_result",
            "recovery_agent_id",
            "chargeback_date",
            "final_settlement",
        ]

        if any(p in norm_col for p in leakage_patterns) and not any(
            w in norm_col for w in ["target", "label", "recovery_status"]
        ):
            if any(
                p in norm_col
                for p in [
                    "actual_recovered",
                    "recovered_amount",
                    "actual_recovered_amount",
                    "recovery_date",
                    "post_recovery",
                    "settled_amount",
                    "final_settlement",
                    "chargeback_date",
                ]
            ):
                return {
                    "status": "WARNING",
                    "reason": "POST_OUTCOME: Field represents data generated during/after recovery process.",
                    "severity": "HIGH",
                }
            elif any(w in norm_col for w in ["actual", "collection_result", "final_status"]):
                return {
                    "status": "WARNING",
                    "reason": "POST_OUTCOME: Future information not available at decision time.",
                    "severity": "HIGH",
                }

        return {"status": "OK", "reason": "", "severity": "NONE"}

    @staticmethod
    def classify_dataset(
        mapped_fields: list[dict[str, Any]], leaked_columns: set = None
    ) -> dict[str, Any]:
        if leaked_columns is None:
            leaked_columns = set()

        reliable = [
            m
            for m in mapped_fields
            if m.get("confidence") in ["HIGH", "MEDIUM", "USER_CONFIRMED"]
            and m.get("original_column") not in leaked_columns
        ]
        detected = set(
            [
                m["canonical_field"]
                for m in reliable
                if m.get("canonical_field") != CanonicalField.UNKNOWN.value
            ]
        )

        has_entity = any(
            f in detected
            for f in [
                CanonicalField.ENTITY_ID.value,
                CanonicalField.ACCOUNT_ID.value,
                CanonicalField.CUSTOMER_ID.value,
            ]
        )
        has_amount = any(
            f in detected for f in [CanonicalField.AMOUNT.value, CanonicalField.BALANCE.value]
        )
        has_time = any(f in detected for f in [CanonicalField.TIMESTAMP.value])
        has_outcome = any(
            f in detected for f in [CanonicalField.OUTCOME.value, CanonicalField.TARGET.value]
        )
        has_tx = CanonicalField.TRANSACTION_ID.value in detected

        missing = []
        if not has_entity:
            missing.append("Entity/Account Identifier")
        if not has_amount:
            missing.append("Monetary/Value Concept")
        if not has_time:
            missing.append("Temporal Concept (Date/Time)")
        if not has_outcome:
            missing.append("Outcome/Target Label")

        diagnostic = {
            "found_concepts": list(detected),
            "missing_concepts": missing,
            "actionable_feedback": "",
            "exclusions": list(leaked_columns),
        }

        if has_entity and has_amount and has_time and has_outcome:
            if leaked_columns:
                classification = DatasetClassification.ANALYSIS_READY
                reason = "Minimum Information Contract satisfied after explicitly excluding leaked variables."
                diagnostic["actionable_feedback"] = (
                    "Ready for descriptive analysis. Leaked variables must be pruned for ML."
                )
            else:
                classification = DatasetClassification.ML_TRAINING_READY
                reason = "Minimum Information Contract satisfied. Contains entity, amount, time, and outcome labels for supervised ML."
                diagnostic["actionable_feedback"] = "Ready for model training."
        elif has_entity and has_amount and has_time:
            classification = DatasetClassification.ANALYSIS_READY
            reason = "Contains core financial dimensions but missing outcome labels for ML."
            diagnostic["actionable_feedback"] = (
                "Upload a dataset with historical outcomes (e.g. payment success/failure) to enable ML."
            )
        elif has_entity or has_amount or has_tx:
            classification = DatasetClassification.PARTIALLY_USABLE
            reason = "Missing core structural requirements (e.g. timestamp or amount) but contains some identifiers."
            diagnostic["actionable_feedback"] = (
                f"Requires missing concepts: {', '.join(missing)} to proceed."
            )
        else:
            classification = DatasetClassification.INSUFFICIENT
            reason = (
                "Does not contain recognizable financial structures or critical data was leaked."
            )
            diagnostic["actionable_feedback"] = (
                "Dataset rejected. Ensure columns contain accounts, amounts, dates, and outcomes."
            )

        return {
            "classification": classification.value,
            "reason": reason,
            "detected_fields": list(detected),
            "diagnostic": diagnostic,
        }


class DatasetProfiler:
    @staticmethod
    def calculate_data_quality_score(df, cols_profile, mapped_schema, leakage_warnings, val_result):
        score = 100.0
        breakdown = []

        # 1. Schema Completeness (Max deduction 60)
        has_id = False
        has_amount = False
        has_target = False
        has_time = False

        for m in mapped_schema:
            cf = m.get("canonical_field")
            if cf in [
                CanonicalField.ENTITY_ID.value,
                CanonicalField.ACCOUNT_ID.value,
                CanonicalField.CUSTOMER_ID.value,
            ]:
                has_id = True
            elif cf in [CanonicalField.AMOUNT.value, CanonicalField.BALANCE.value]:
                has_amount = True
            elif cf in [CanonicalField.TARGET.value, CanonicalField.OUTCOME.value]:
                has_target = True
            elif cf == CanonicalField.TIMESTAMP.value:
                has_time = True

        if not has_id:
            score -= 15.0
            breakdown.append("-15.0: Missing Entity Identifier")
        if not has_amount:
            score -= 15.0
            breakdown.append("-15.0: Missing Amount/Value column")
        if not has_target:
            score -= 15.0
            breakdown.append("-15.0: Missing Target/Outcome column")
        if not has_time:
            score -= 15.0
            breakdown.append("-15.0: Missing Timestamp column")

        # 2. Missingness
        total_cells = df.size
        if total_cells > 0:
            missing_rate = int(df.isnull().sum().sum()) / total_cells
            if missing_rate > 0.05:
                deduct = min(30.0, (missing_rate - 0.05) * 100.0)
                score -= deduct
                breakdown.append(
                    f"-{deduct:.1f}: Overall missing value rate is {missing_rate * 100:.1f}%"
                )

        # 3. Duplicates
        if len(df) > 0:
            dup_cnt = int(df.duplicated().sum())
            dup_rate = dup_cnt / len(df)
            if dup_rate > 0.01:
                deduct = min(20.0, dup_rate * 100.0)
                score -= deduct
                breakdown.append(f"-{deduct:.1f}: Duplicate rows ({dup_rate * 100:.1f}%)")

        # 4. Leakage
        if leakage_warnings:
            deduct = min(30.0, len(leakage_warnings) * 10.0)
            score -= deduct
            breakdown.append(f"-{deduct:.1f}: {len(leakage_warnings)} leakage warning(s) detected")

        # 5. Constant columns
        const_cols = [c for c in cols_profile if c.get("is_constant")]
        if const_cols:
            deduct = min(10.0, len(const_cols) * 2.0)
            score -= deduct
            breakdown.append(f"-{deduct:.1f}: {len(const_cols)} constant column(s)")

        # 6. Target class imbalance
        target_col = None
        for m in mapped_schema:
            if (
                m.get("canonical_field")
                in [CanonicalField.TARGET.value, CanonicalField.OUTCOME.value]
                and m.get("confidence") != "LOW"
            ):
                target_col = m.get("original_column")
                break

        if target_col and target_col in df.columns:
            counts = df[target_col].value_counts(normalize=True)
            if not counts.empty:
                min_class = counts.min()
                if min_class < 0.01:
                    score -= 15.0
                    breakdown.append(
                        f"-15.0: Extreme class imbalance (minority class {min_class * 100:.2f}%)"
                    )

        final_score = max(0.0, min(100.0, score))
        if final_score == 100.0:
            breakdown.append("+0.0: Perfect data quality score")

        return {"score": float(round(final_score, 1)), "breakdown": breakdown}

    @staticmethod
    def profile_file(file_path: str, file_type: str) -> dict[str, Any]:
        row_count = 0
        duplicate_count = 0
        total_missing_values = 0

        try:
            if file_path.endswith(".parquet") or file_type == "application/parquet":
                import pyarrow.parquet as pq

                pf = pq.ParquetFile(file_path)
                row_count = pf.metadata.num_rows
                df = pf.read_row_group(0).to_pandas()
                if len(df) > 5000:
                    df = df.head(5000)
                duplicate_count = int(df.duplicated().sum() * (row_count / max(1, len(df))))
                total_missing_values = int(df.isnull().sum().sum() * (row_count / max(1, len(df))))

            elif file_path.endswith(".csv") or file_type == "text/csv":
                df = pd.read_csv(file_path, nrows=5000)
                with open(file_path, encoding="utf-8", errors="ignore") as f_in:
                    row_count = sum(1 for _ in f_in) - 1
                    if row_count < 0:
                        row_count = 0
                duplicate_count = int(df.duplicated().sum() * (row_count / max(1, len(df))))
                total_missing_values = int(df.isnull().sum().sum() * (row_count / max(1, len(df))))

            elif file_path.lower().endswith(".zip") or file_type == "zip":
                # A .zip holding exactly one CSV. Upload accepts .zip, but it used to fall
                # into the Excel branch below and fail with the meaningless "No such
                # keys(s): 'io.excel.zip.reader'". An archive with several members now
                # fails with pandas' own clear "Multiple files found in ZIP file".
                df = pd.read_csv(file_path, nrows=5000, compression="zip")
                row_count = sum(
                    len(chunk)
                    for chunk in pd.read_csv(
                        file_path, compression="zip", usecols=[0], chunksize=100_000
                    )
                )
                duplicate_count = int(df.duplicated().sum() * (row_count / max(1, len(df))))
                total_missing_values = int(df.isnull().sum().sum() * (row_count / max(1, len(df))))

            else:
                df = pd.read_excel(file_path, nrows=5000)
                try:
                    import openpyxl

                    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
                    ws = wb.active
                    row_count = max(0, ws.max_row - 1)
                    wb.close()
                except Exception:
                    row_count = len(df)
                duplicate_count = int(df.duplicated().sum() * (row_count / max(1, len(df))))
                total_missing_values = int(df.isnull().sum().sum() * (row_count / max(1, len(df))))

        except Exception as e:
            raise ValueError(f"Failed to read dataset: {e}") from e

        col_count = len(df.columns)

        mapper = SemanticMapper()
        cols_profile = []
        leakage_warnings = []
        constant_columns = []
        leaked_columns = set()

        mapped_schema = mapper.map_schema(df)

        for mapped in mapped_schema:
            col = mapped["original_column"]
            series = df[col]
            dtype = str(series.dtype)
            unique_cnt = int(series.nunique())
            null_cnt = int(series.isnull().sum())

            if unique_cnt <= 1:
                constant_columns.append(col)

            mapped["role"] = (
                "FEATURE"
                if mapped["canonical_field"]
                not in [
                    CanonicalField.OUTCOME.value,
                    CanonicalField.TARGET.value,
                    CanonicalField.UNKNOWN.value,
                ]
                else "TARGET"
                if mapped["canonical_field"] != CanonicalField.UNKNOWN.value
                else "UNKNOWN"
            )
            mapped["feature_eligible"] = True
            mapped["exclusion_reason"] = ""

            leakage = DatasetValidator.detect_leakage(col, mapped["canonical_field"])
            if leakage["status"] == "WARNING":
                leakage_warnings.append(
                    {"column": col, "reason": leakage["reason"], "severity": leakage["severity"]}
                )
                leaked_columns.add(col)
                mapped["feature_eligible"] = False
                mapped["exclusion_reason"] = leakage["reason"]

            cols_profile.append(
                {
                    "column_name": col,
                    "dtype": dtype,
                    "missing_count": int(null_cnt * (row_count / max(1, len(df)))),
                    "missing_rate": null_cnt / max(1, len(df)),
                    "unique_count": unique_cnt,
                    "is_constant": unique_cnt <= 1,
                }
            )

        val_result = DatasetValidator.classify_dataset(mapped_schema, leaked_columns)

        return {
            "row_count": row_count,
            "column_count": col_count,
            "duplicate_row_count": duplicate_count,
            "total_missing_values": total_missing_values,
            "columns_profile": cols_profile,
            "leakage_warnings": leakage_warnings,
            "mapped_schema": mapped_schema,
            "data_quality_score": DatasetProfiler.calculate_data_quality_score(
                df, cols_profile, mapped_schema, leakage_warnings, val_result
            ),
            "validation": val_result,
        }
