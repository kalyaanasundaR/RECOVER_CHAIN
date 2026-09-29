"""Phase 11 — drift-readiness infrastructure.

This is READINESS infrastructure, not a live production-drift monitor: RecoverChain
has no production inference traffic to monitor yet (see
docs/history/PHASE_10_ML_DL_EVALUATION_REPORT.md). What this module gives is the
one thing that was missing — a reusable, dependency-free way to compare two feature
distributions (e.g. training vs. test, or a trained model's training snapshot vs. a
newly ingested dataset) and flag columns whose distribution has shifted enough to be
worth a human look before trusting a model on the new data.

Two building blocks:
  - `summarize_distribution(df, columns)` — a per-column summary (numeric: mean,
    std, quantiles; categorical: top value frequencies), computed once and cheap to
    store in registry metadata.
  - `compare_distributions(baseline, candidate)` — takes two such summaries and
    returns a per-column drift score plus an overall flag. Numeric columns use a
    standardized mean shift (in pooled-std units); categorical columns use total
    variation distance between the two frequency tables. Both are bounded, simple,
    and don't require scipy — genuinely lightweight, not a placeholder for a real
    drift library RecoverChain doesn't have installed.
"""

from __future__ import annotations

import pandas as pd

# A standardized mean shift beyond this many pooled-std units, or a categorical
# total-variation-distance beyond this fraction, is flagged as DRIFTED. These are
# deliberately conservative defaults for a readiness tool, not a tuned production
# alerting threshold (no production traffic exists yet to tune against).
NUMERIC_DRIFT_THRESHOLD = 0.5
CATEGORICAL_DRIFT_THRESHOLD = 0.3


def summarize_distribution(df: pd.DataFrame, columns: list[str]) -> dict:
    """One summary per column: numeric columns get mean/std/min/max/quantiles;
    everything else gets the top-10 value frequencies (as fractions)."""
    summary: dict[str, dict] = {}
    for col in columns:
        if col not in df.columns:
            summary[col] = {"type": "MISSING"}
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            clean = pd.to_numeric(series, errors="coerce").dropna()
            if clean.empty:
                summary[col] = {"type": "numeric", "count": 0}
                continue
            summary[col] = {
                "type": "numeric",
                "count": int(clean.count()),
                "mean": float(clean.mean()),
                "std": float(clean.std(ddof=0)) if clean.count() > 1 else 0.0,
                "min": float(clean.min()),
                "max": float(clean.max()),
                "p50": float(clean.quantile(0.5)),
            }
        else:
            clean = series.astype(str).replace({"nan": None, "None": None}).dropna()
            total = len(clean)
            if total == 0:
                summary[col] = {"type": "categorical", "count": 0, "frequencies": {}}
                continue
            counts = clean.value_counts(normalize=True).head(10)
            summary[col] = {
                "type": "categorical",
                "count": total,
                "frequencies": {str(k): float(v) for k, v in counts.items()},
            }
    return summary


def compare_distributions(baseline: dict, candidate: dict) -> dict:
    """Compares two `summarize_distribution` outputs column by column. Returns
    per-column {drift_score, drifted, reason} plus an overall `any_drifted` flag.
    Columns present in only one side, or with too little data, are reported as
    UNKNOWN rather than silently skipped or scored as zero drift."""
    columns = sorted(set(baseline) | set(candidate))
    per_column: dict[str, dict] = {}
    any_drifted = False

    for col in columns:
        b = baseline.get(col)
        c = candidate.get(col)
        if (
            not b
            or not c
            or b.get("type") in (None, "MISSING")
            or c.get("type") in (None, "MISSING")
        ):
            per_column[col] = {
                "drift_score": None,
                "drifted": False,
                "reason": "UNKNOWN — column missing on one side",
            }
            continue
        if b["type"] != c["type"]:
            per_column[col] = {
                "drift_score": None,
                "drifted": True,
                "reason": f"type changed: {b['type']} -> {c['type']}",
            }
            any_drifted = True
            continue

        if b["type"] == "numeric":
            if b.get("count", 0) == 0 or c.get("count", 0) == 0:
                per_column[col] = {
                    "drift_score": None,
                    "drifted": False,
                    "reason": "UNKNOWN — insufficient data",
                }
                continue
            pooled_std = ((b["std"] ** 2 + c["std"] ** 2) / 2) ** 0.5
            if pooled_std == 0:
                # No spread on either side: any mean difference is a full, not partial,
                # shift. 1e6 stands in for "unbounded" while staying JSON-serializable
                # (float('inf') is not valid JSON).
                score = 0.0 if b["mean"] == c["mean"] else 1e6
            else:
                score = abs(b["mean"] - c["mean"]) / pooled_std
            drifted = score > NUMERIC_DRIFT_THRESHOLD
            per_column[col] = {
                "drift_score": score,
                "drifted": drifted,
                "reason": f"standardized mean shift {score:.3f} (threshold {NUMERIC_DRIFT_THRESHOLD})",
            }
        else:
            keys = set(b.get("frequencies", {})) | set(c.get("frequencies", {}))
            tvd = 0.5 * sum(
                abs(b.get("frequencies", {}).get(k, 0.0) - c.get("frequencies", {}).get(k, 0.0))
                for k in keys
            )
            drifted = tvd > CATEGORICAL_DRIFT_THRESHOLD
            per_column[col] = {
                "drift_score": tvd,
                "drifted": drifted,
                "reason": f"total variation distance {tvd:.3f} (threshold {CATEGORICAL_DRIFT_THRESHOLD})",
            }

        any_drifted = any_drifted or per_column[col]["drifted"]

    return {"any_drifted": any_drifted, "columns": per_column}
