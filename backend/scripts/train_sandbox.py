"""Run MLTrainingEngine standalone, outside the API/DB — a sandbox for experimenting
with the shadow recovery-probability model. Writes its artifacts to their own
directory so it never touches the app's real model registry.

Usage:
    venv/Scripts/python.exe scripts/train_sandbox.py <path-to-csv-or-xlsx> \
        --target status --features amount invoice_date reason \
        [--prediction-problem payment-failure-risk] [--min-roc-auc 0.55] [--min-test-rows 15]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on path

from application.ml_training import MLTrainingEngine  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_path", help="CSV / XLSX / Parquet file to train on")
    p.add_argument("--target", required=True, help="Column holding the success/failure label")
    p.add_argument("--features", nargs="+", required=True, help="Feature columns to train on")
    p.add_argument("--prediction-problem", default="payment-failure-risk")
    p.add_argument("--dataset-id", default="sandbox")
    p.add_argument("--min-roc-auc", type=float, default=0.55)
    p.add_argument("--min-test-rows", type=int, default=15)
    p.add_argument("--out", default="scripts/_sandbox_output", help="Where artifacts go")
    args = p.parse_args()

    # This dict is exactly what MLReadinessAnalyzer.analyze_readiness() would
    # normally build from a DB-backed dataset row (dataset_router.py calls it
    # in /{dataset_id}/ml-readiness). Only target_column and feature_columns
    # are required by the engine itself; everything else has a safe default.
    spec = {
        "dataset_id": args.dataset_id,
        "target_column": args.target,
        "feature_columns": args.features,
        "prediction_problem": args.prediction_problem,
        "target_definition": "Sandbox run — not the app's real spec",
        "dataset_provenance_status": "SANDBOX",
        # "temporal_split": {"split_column": "invoice_date"},  # optional; omit to
        # fall back to a stratified random split (see ml_training.py _split_data)
    }

    os.makedirs(args.out, exist_ok=True)
    engine = MLTrainingEngine(
        spec,
        data_path=args.data_path,
        output_dir=args.out,
        min_roc_auc=args.min_roc_auc,
        min_test_rows=args.min_test_rows,
    )
    metadata = engine.train_and_evaluate()

    print(json.dumps(metadata.get("quality_gate"), indent=2))
    print(json.dumps(metadata.get("final_test_metrics"), indent=2))
    print(f"selected_model = {metadata['selected_model']}  status = {metadata['status']}")
    print(f"artifact saved to {metadata['artifact_path']}")


if __name__ == "__main__":
    main()
