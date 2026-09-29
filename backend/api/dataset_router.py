import json
import os
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.auth import verify_api_key
from application.dataset_lab import DatasetLabService
from domain.models import RiskCategory
from infrastructure.dataset_orm import DatasetMetadataModel, DatasetStatus
from infrastructure.db import get_db

# Every state-changing / compute-triggering route in this router is guarded;
# read-only GETs stay open for the dashboard.
_AUTH = [Depends(verify_api_key)]

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _registry_dir() -> str:
    """The model registry folder (configurable; see infrastructure/storage.py)."""
    from infrastructure.storage import model_registry_dir

    return model_registry_dir()


def _delete_models_for_dataset(dataset_id: str) -> int:
    """Remove every model artifact + metadata file belonging to a dataset."""
    removed = 0
    if not os.path.isdir(_registry_dir()):
        return 0
    for f in list(os.listdir(_registry_dir())):
        if not f.endswith("_metadata.json"):
            continue
        path = os.path.join(_registry_dir(), f)
        try:
            meta = json.load(open(path))
        except Exception:
            continue
        if meta.get("dataset_id") != dataset_id:
            continue
        model_id = meta.get("model_id") or f[: -len("_metadata.json")]
        for name in (f, f"{model_id}_model.joblib"):
            p = os.path.join(_registry_dir(), name)
            try:
                if os.path.exists(p):
                    os.remove(p)
                    removed += 1
            except OSError:
                pass
    return removed


def _prune_old_models_for_dataset(dataset_id: str, keep_model_id: str = None) -> int:
    """Delete every model for `dataset_id` except the one identified by
    `keep_model_id` (the freshly trained one)."""
    removed = 0
    if not os.path.isdir(_registry_dir()):
        return 0
    for f in list(os.listdir(_registry_dir())):
        if not f.endswith("_metadata.json"):
            continue
        path = os.path.join(_registry_dir(), f)
        try:
            meta = json.load(open(path))
        except Exception:
            continue
        if meta.get("dataset_id") != dataset_id:
            continue
        mid = meta.get("model_id") or f[: -len("_metadata.json")]
        if keep_model_id and mid == keep_model_id:
            continue
        for name in (f, f"{mid}_model.joblib"):
            p = os.path.join(_registry_dir(), name)
            try:
                if os.path.exists(p):
                    os.remove(p)
                    removed += 1
            except OSError:
                pass
    return removed


def _prune_stale_datasets(db: Session, keep: int) -> dict:
    """Keep the `keep` most recently uploaded datasets; delete the rest — their
    metadata row, uploaded file, and model artifacts. Generated recovery cases
    are left intact (they are the history shown in Insights)."""
    keep = max(1, min(keep, 200))
    svc = DatasetLabService(db)
    rows = sorted(
        svc.get_all_datasets(), key=lambda d: d.upload_timestamp or datetime.min, reverse=True
    )
    stale = rows[keep:]
    files = models = 0
    ids = []
    for ds in stale:
        try:
            fp = os.path.join(svc.dataset_dir, ds.filename)
            if os.path.exists(fp):
                os.remove(fp)
                files += 1
        except OSError:
            pass
        models += _delete_models_for_dataset(ds.dataset_id)
        db.delete(ds)
        ids.append(ds.dataset_id)
    if ids:
        db.commit()

    # Sweep model files whose dataset no longer exists at all (e.g. left behind
    # by tests or earlier prunes).
    orphans = 0
    live_ids = {d.dataset_id for d in svc.get_all_datasets()}
    if os.path.isdir(_registry_dir()):
        for f in list(os.listdir(_registry_dir())):
            if not f.endswith("_metadata.json"):
                continue
            try:
                meta = json.load(open(os.path.join(_registry_dir(), f)))
            except Exception:
                continue
            did = meta.get("dataset_id")
            if did and did not in live_ids and did != "legacy_billing_v3":
                mid = meta.get("model_id") or f[: -len("_metadata.json")]
                for name in (f, f"{mid}_model.joblib"):
                    p = os.path.join(_registry_dir(), name)
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                            orphans += 1
                    except OSError:
                        pass

    return {
        "removed_datasets": len(ids),
        "removed_files": files,
        "removed_model_files": models + orphans,
        "dataset_ids": ids,
    }


@router.post("/prune", dependencies=_AUTH)
def prune_datasets(keep: int = 8, db: Session = Depends(get_db)):
    return {"status": "success", **_prune_stale_datasets(db, keep)}


@router.post("/sync", dependencies=_AUTH)
def sync_local_datasets(db: Session = Depends(get_db)):
    service = DatasetLabService(db)
    imported = service.import_local_datasets()
    return {"status": "success", "imported_count": len(imported), "dataset_ids": imported}


from api.security import MAX_UPLOAD_BYTES as MAX_FILE_SIZE  # configurable via MAX_UPLOAD_BYTES env


@router.post("/upload", dependencies=_AUTH)
def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    service = DatasetLabService(db)

    # Security: validate extension
    if not file.filename or "." not in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    ext = file.filename.split(".")[-1].lower()
    if ext not in ["csv", "parquet", "xlsx", "zip"]:
        raise HTTPException(
            status_code=400, detail="Unsupported file format. Must be CSV, Parquet, or XLSX."
        )

    # Sanitize filename strictly to prevent path traversal
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in [".", "_", "-"])
    if not safe_filename or safe_filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid safe filename.")

    # Store under a unique, dataset-scoped name so re-uploading the same file
    # never collides. The display name keeps the original.
    ds_id = f"ds_{uuid.uuid4().hex[:8]}"
    stored_filename = f"{ds_id}_{safe_filename}"
    os.makedirs(service.dataset_dir, exist_ok=True)
    file_path = os.path.join(service.dataset_dir, stored_filename)

    # Chunked write with size limit; the bytes are hashed on the way through.
    import hashlib

    hasher = hashlib.sha256()
    bytes_written = 0
    oversize = False
    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = file.file.read(8192)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE:
                    oversize = True
                    break
                buffer.write(chunk)
                hasher.update(chunk)
    finally:
        file.file.close()

    if oversize:
        os.remove(file_path)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MiB.",
        )

    size = os.path.getsize(file_path)
    content_sha256 = hasher.hexdigest()
    # A byte-identical earlier upload. The upload still succeeds - replaying it is safe,
    # because file events are deduplicated by content - but the caller is told.
    prior = (
        db.query(DatasetMetadataModel)
        .filter(DatasetMetadataModel.content_sha256 == content_sha256)
        .order_by(DatasetMetadataModel.upload_timestamp.asc())
        .first()
    )
    duplicate_of = prior.dataset_id if prior else None
    new_ds = DatasetMetadataModel(
        dataset_id=ds_id,
        name=safe_filename,
        filename=stored_filename,
        file_type=ext,
        file_size_bytes=size,
        content_sha256=content_sha256,
        upload_timestamp=datetime.now(UTC),
        status=DatasetStatus.PENDING,
    )
    db.add(new_ds)
    db.commit()

    # Keep the workspace focused on recent uploads: drop older datasets (their
    # metadata row, file, and trained models). Recovery cases stay in Insights.
    try:
        pruned = _prune_stale_datasets(db, keep=int(os.getenv("DATASET_KEEP", "12")))
        if pruned["removed_datasets"]:
            print(
                f"[upload] pruned {pruned['removed_datasets']} old dataset(s), "
                f"{pruned['removed_model_files']} model file(s)"
            )
    except Exception as e:
        print(f"[upload] prune skipped: {e}")

    return {
        "status": "success",
        "dataset_id": ds_id,
        "content_sha256": content_sha256,
        "duplicate_of": duplicate_of,
    }


def serialize_dataset(ds):
    d = ds.__dict__.copy()
    d.pop("_sa_instance_state", None)
    return d


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    service = DatasetLabService(db)
    datasets = service.get_all_datasets()
    return {"datasets": [serialize_dataset(ds) for ds in datasets]}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    service = DatasetLabService(db)
    ds = service.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return serialize_dataset(ds)


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: str, limit: int = 25, db: Session = Depends(get_db)):
    limit = min(max(1, limit), 100)
    service = DatasetLabService(db)
    ds = service.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    path = os.path.join(service.dataset_dir, ds.filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset file missing")

    import pandas as pd

    _pl = path.lower()
    try:
        if _pl.endswith(".parquet"):
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            df = pf.read_row_group(0).to_pandas().head(limit)
        elif _pl.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path, nrows=limit)
        else:
            df = pd.read_csv(path, nrows=limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate preview: {e}") from e

    schema_map = {
        s["original_column"]: s["canonical_field"] for s in (ds.recoverchain_signals or [])
    }

    columns = []
    for col in df.columns:
        columns.append(
            {"original_name": str(col), "canonical_concept": schema_map.get(col, "UNKNOWN")}
        )

    # Convert dataframe to JSON serializable records
    raw_records = df.to_dict(orient="records")
    records = []
    for r in raw_records:
        clean_r = {}
        for k, v in r.items():
            clean_r[k] = None if pd.isna(v) else v
        records.append(clean_r)
    return {
        "dataset_id": dataset_id,
        "row_count_preview": len(records),
        "columns": columns,
        "rows": records,
    }


@router.post("/{dataset_id}/analyze", dependencies=_AUTH)
def analyze_dataset(dataset_id: str, db: Session = Depends(get_db)):
    service = DatasetLabService(db)
    ds = service.analyze_dataset(dataset_id)
    return {"status": ds.status.value, "dataset_id": ds.dataset_id}


@router.post("/{dataset_id}/ml-readiness", dependencies=_AUTH)
def evaluate_ml_readiness(dataset_id: str, db: Session = Depends(get_db)):
    service = DatasetLabService(db)
    ds = service.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if ds.status not in [
        DatasetStatus.COMPLETED,
        DatasetStatus.READY_FOR_ANALYSIS,
        DatasetStatus.FAILED,
    ]:
        raise HTTPException(status_code=400, detail="Dataset must have confirmed mappings first.")

    ds.status = DatasetStatus.ANALYZING
    db.commit()

    from application.ml_readiness import MLReadinessAnalyzer

    file_path = os.path.join(service.dataset_dir, ds.filename)
    try:
        spec = MLReadinessAnalyzer.analyze_readiness(ds, file_path)
        ds.training_suitability = spec
        if spec.get("readiness_status") in [
            "ML_TRAINING_READY",
            "ML_TRAINING_READY_WITH_EXCLUSIONS",
            "ML_TRAINING_READY_WITH_WARNINGS",
        ]:
            ds.status = DatasetStatus.ML_READY
        else:
            ds.status = DatasetStatus.FAILED
        db.commit()
        return spec
    except Exception as e:
        ds.status = DatasetStatus.FAILED
        ds.error_message = str(e)
        db.commit()
        raise HTTPException(status_code=400, detail=f"ML readiness evaluation failed: {e}") from e


@router.get("/{dataset_id}/ml-readiness")
def get_ml_readiness(dataset_id: str, db: Session = Depends(get_db)):
    service = DatasetLabService(db)
    ds = service.get_dataset(dataset_id)
    if not ds or not ds.training_suitability:
        raise HTTPException(status_code=404, detail="Readiness spec not found")
    return ds.training_suitability


def run_ml_training_task(
    dataset_id: str,
    spec: dict,
    data_path: str,
    min_roc_auc: float = None,
    min_test_rows: int = None,
):
    from application.ml_training import MLTrainingEngine
    from infrastructure.dataset_orm import DatasetMetadataModel, DatasetStatus
    from infrastructure.db import SessionLocal

    db = SessionLocal()
    try:
        engine = MLTrainingEngine(
            spec,
            data_path,
            _registry_dir(),
            min_roc_auc=min_roc_auc,
            min_test_rows=min_test_rows,
        )
        meta = engine.train_and_evaluate()

        # Keep only the newest model per dataset — drop older / rejected ones so
        # the registry does not grow without bound.
        try:
            _prune_old_models_for_dataset(dataset_id, keep_model_id=meta.get("model_id"))
        except Exception:
            pass

        ds = (
            db.query(DatasetMetadataModel)
            .filter(DatasetMetadataModel.dataset_id == dataset_id)
            .first()
        )
        if ds:
            ds.status = DatasetStatus.TRAINED
            db.commit()
    except Exception as e:
        print(f"Training failed for {dataset_id}: {e}")
        ds = (
            db.query(DatasetMetadataModel)
            .filter(DatasetMetadataModel.dataset_id == dataset_id)
            .first()
        )
        if ds:
            ds.status = DatasetStatus.FAILED
            ds.error_message = str(e)
            db.commit()
    finally:
        db.close()


@router.post("/{dataset_id}/train", dependencies=_AUTH)
def start_training(
    dataset_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    service = DatasetLabService(db)
    ds = service.get_dataset(dataset_id)
    if not ds or not ds.training_suitability:
        raise HTTPException(status_code=400, detail="Dataset must have ML Readiness spec first.")

    if ds.status not in [DatasetStatus.ML_READY, DatasetStatus.COMPLETED, DatasetStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Dataset must be ML ready first.")

    spec = ds.training_suitability
    status = spec.get("readiness_status", "")

    # Quality Gate
    if status not in [
        "ML_TRAINING_READY",
        "ML_TRAINING_READY_WITH_EXCLUSIONS",
        "ML_TRAINING_READY_WITH_WARNINGS",
    ]:
        raise HTTPException(
            status_code=400, detail=f"Dataset not ready for training. Status: {status}"
        )

    ds.status = DatasetStatus.TRAINING
    db.commit()

    file_path = os.path.join(service.dataset_dir, ds.filename)
    background_tasks.add_task(run_ml_training_task, dataset_id, spec, file_path)
    return {"status": "Training initiated"}


@router.get("/{dataset_id}/models")
def list_trained_models(dataset_id: str):
    registry_dir = _registry_dir()
    if not os.path.exists(registry_dir):
        return []

    models = []
    for f in os.listdir(registry_dir):
        if f.endswith("_metadata.json"):
            try:
                with open(os.path.join(registry_dir, f)) as meta_f:
                    import json

                    meta = json.load(meta_f)
                    if meta.get("dataset_id") == dataset_id:
                        models.append(meta)
            except Exception:
                continue

    # Sort by created_at descending
    models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return models


@router.get("/{dataset_id}/workflow-status")
def get_workflow_status(dataset_id: str, db: Session = Depends(get_db)):
    service = DatasetLabService(db)
    ds = service.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    detected_fields = ds.recoverchain_signals or []

    # Minimum-information contract: each slot below is satisfied by any of its aliases.
    from application.dataset_intelligence import CanonicalField

    detected_canonical = [f["canonical_field"] for f in detected_fields if f.get("canonical_field")]

    # Check what is missing (handle aliases in contract)
    missing_required_fields = []
    has_entity = any(
        f in detected_canonical
        for f in [
            CanonicalField.ENTITY_ID.value,
            CanonicalField.ACCOUNT_ID.value,
            CanonicalField.CUSTOMER_ID.value,
        ]
    )
    if not has_entity:
        missing_required_fields.append("ENTITY_ID/ACCOUNT_ID")

    has_amount = any(
        f in detected_canonical for f in [CanonicalField.AMOUNT.value, CanonicalField.BALANCE.value]
    )
    if not has_amount:
        missing_required_fields.append("AMOUNT/BALANCE")

    has_time = any(
        f in detected_canonical
        for f in [CanonicalField.TIMESTAMP.value, CanonicalField.SETTLEMENT_DATE.value]
    )
    if not has_time:
        missing_required_fields.append("TIMESTAMP/DATE")

    has_target = any(
        f in detected_canonical for f in [CanonicalField.TARGET.value, CanonicalField.OUTCOME.value]
    )
    if not has_target:
        missing_required_fields.append("TARGET/OUTCOME")

    quality_stats = ds.data_quality_report or {}
    leakage_warnings = ds.leakage_detection or []

    suitability = ds.training_suitability or {}
    target_classification = suitability.get("prediction_problem", "unknown")
    ml_readiness = suitability.get("readiness_status", "PENDING" if not suitability else "UNKNOWN")
    if not suitability and ds.status == DatasetStatus.MAPPING_REVIEW:
        ml_readiness = "AWAITING_MAPPING_REVIEW"

    return {
        "dataset_id": ds.dataset_id,
        "filename": ds.filename,
        "status": ds.status.value,
        "row_count": ds.row_count,
        "column_count": ds.column_count,
        "detected_canonical_fields": detected_fields,
        "required_fields": [
            "ENTITY_ID/ACCOUNT_ID",
            "AMOUNT/BALANCE",
            "TIMESTAMP/DATE",
            "TARGET/OUTCOME",
        ],
        "missing_required_fields": missing_required_fields,
        "quality_statistics": quality_stats,
        "leakage_warnings": leakage_warnings,
        "target_classification": target_classification,
        "ml_readiness_status": ml_readiness,
    }


from pydantic import BaseModel


class MappingOverride(BaseModel):
    original_column: str
    canonical_field: str
    action: str  # "confirm", "override", "unused"


class MappingConfirmationRequest(BaseModel):
    mappings: list[MappingOverride]
    # Dataset category contract (Phase 4). Omitted -> what is stored stays unchanged;
    # an explicit null clears a declared domain.
    declared_domain: RiskCategory | None = None
    source_system: str | None = None


class DatasetDomainRequest(BaseModel):
    declared_domain: RiskCategory | None = None  # null clears a declaration
    source_system: str | None = None


# Rows read to classify a dataset's domain outside a case run (the same bound as a run).
_CLASSIFICATION_SAMPLE_ROWS = 500


def _read_source_frame(file_path: str, nrows: int):
    """The first ``nrows`` rows of a dataset file, and the file's total row count when
    the format reports it without a full read (parquet), else None."""
    import pandas as pd

    fpl = file_path.lower()
    if fpl.endswith(".parquet"):
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(file_path)
        return pf.read_row_group(0).to_pandas().head(nrows), pf.metadata.num_rows
    if fpl.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path, nrows=nrows), None
    return pd.read_csv(file_path, nrows=nrows), None


def _category_column(mappings) -> str | None:
    return next(
        (
            m["original_column"]
            for m in (mappings or [])
            if m.get("canonical_field") == "EVENT_CATEGORY"
        ),
        None,
    )


def _classify(ds, df):
    from application.domain_classification import classify_dataset

    return classify_dataset(
        df,
        declared_domain=ds.declared_domain,
        category_column=_category_column(ds.recoverchain_signals),
    )


def _sample_classification(service: DatasetLabService, ds):
    """Classify from a bounded sample of the file; None when the file cannot be read."""
    path = os.path.join(service.dataset_dir, ds.filename)
    if not os.path.exists(path):
        return None
    try:
        df, _ = _read_source_frame(path, _CLASSIFICATION_SAMPLE_ROWS)
    except Exception:
        return None
    return _classify(ds, df)


def _record_domain_declaration(db: Session, ds, declared: str | None, via: str) -> None:
    if declared == ds.declared_domain:
        return
    from infrastructure.repositories import SqlAlchemyAuditRecorder

    SqlAlchemyAuditRecorder(db).log_transition(
        case_id="SYSTEM",
        from_state="N/A",
        to_state="N/A",
        evidence={
            "action": "dataset_domain_declared",
            "dataset_id": ds.dataset_id,
            "declared_domain": declared,
            "previous_declared_domain": ds.declared_domain,
            "via": via,
        },
    )
    ds.declared_domain = declared


def _domain_contract(ds, classification) -> dict:
    from domain.revenue_domains import DOMAIN_SPECS

    return {
        "dataset_id": ds.dataset_id,
        "declared_domain": ds.declared_domain,
        "source_system": ds.source_system,
        "schema": [
            {"original_column": m["original_column"], "canonical_field": m["canonical_field"]}
            for m in (ds.recoverchain_signals or [])
        ],
        "content_sha256": ds.content_sha256,  # the dataset version
        "domain_classification": classification.to_dict()
        if classification is not None
        else ds.domain_classification,
        "domains": [
            {"domain": c.value, "canonical_name": s.canonical_name, "label": s.label}
            for c, s in DOMAIN_SPECS.items()
        ],
    }


@router.get("/{dataset_id}/domain")
def get_dataset_domain(dataset_id: str, db: Session = Depends(get_db)):
    """The dataset category contract: declared domain, source, schema, version, and the
    classification the current rules produce for the file (recomputed; nothing stored)."""
    service = DatasetLabService(db)
    ds = service.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return _domain_contract(ds, _sample_classification(service, ds))


@router.put("/{dataset_id}/domain", dependencies=_AUTH)
def declare_dataset_domain(
    dataset_id: str, req: DatasetDomainRequest, db: Session = Depends(get_db)
):
    """Declare (or clear) the dataset's domain. A declared domain is used as given for
    every row; the declaration is audited."""
    service = DatasetLabService(db)
    ds = service.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    _record_domain_declaration(
        db, ds, req.declared_domain.value if req.declared_domain else None, "PUT /domain"
    )
    if "source_system" in req.model_fields_set:
        ds.source_system = (req.source_system or "").strip() or None
    classification = _sample_classification(service, ds)
    ds.domain_classification = classification.to_dict() if classification is not None else None
    db.commit()
    return _domain_contract(ds, classification)


@router.post("/{dataset_id}/mapping", dependencies=_AUTH)
def confirm_dataset_mapping(
    dataset_id: str, req: MappingConfirmationRequest, db: Session = Depends(get_db)
):
    service = DatasetLabService(db)
    ds = service.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    profile = ds.columns_profile or []
    valid_columns = {c["column_name"] for c in profile}

    # Validation 1: Nonexistent columns
    for m in req.mappings:
        if m.original_column not in valid_columns:
            raise HTTPException(
                status_code=400, detail=f"Column {m.original_column} does not exist."
            )

    # Compile final mappings
    final_mappings = []
    canonical_used = set()

    leakage = ds.leakage_detection or []
    leaked_cols = {l["column"] for l in leakage}

    from application.dataset_intelligence import CanonicalField

    for m in req.mappings:
        if m.action == "unused" or m.canonical_field == CanonicalField.UNKNOWN.value:
            final_mappings.append(
                {
                    "original_column": m.original_column,
                    "canonical_field": CanonicalField.UNKNOWN.value,
                    "confidence": "USER_CONFIRMED",
                    "mapping_reason": "User explicitly marked as unused or unknown.",
                }
            )
            continue

        # Validation 2: Duplicate assignments for single-use fields
        single_use = [
            CanonicalField.TARGET.value,
            CanonicalField.OUTCOME.value,
            CanonicalField.AMOUNT.value,
            CanonicalField.EVENT_CATEGORY.value,
        ]
        if m.canonical_field in single_use:
            if m.canonical_field in canonical_used:
                raise HTTPException(
                    status_code=400,
                    detail=f"Duplicate assignment for {m.canonical_field} prohibited.",
                )
            canonical_used.add(m.canonical_field)

        # Validation 3: Target mapped to post-outcome
        if m.canonical_field in [CanonicalField.TARGET.value, CanonicalField.OUTCOME.value]:
            if m.original_column in leaked_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot map target to leaked/post-outcome field: {m.original_column}",
                )

        final_mappings.append(
            {
                "original_column": m.original_column,
                "canonical_field": m.canonical_field,
                "confidence": "USER_CONFIRMED",
                "mapping_reason": "User confirmed override.",
            }
        )

    # Validation 4: LOW confidence rejection / unsafe mappings
    # Evaluate against DatasetValidator
    from application.dataset_intelligence import DatasetValidator

    validation_res = DatasetValidator.classify_dataset(final_mappings, leaked_columns=leaked_cols)
    if validation_res["classification"] in ["INSUFFICIENT", "PARTIALLY_USABLE"]:
        raise HTTPException(status_code=400, detail=f"Unsafe mappings: {validation_res['reason']}")

    # Validation 5: LOW confidence critical fields must be explicitly overridden?
    # If the user provides overrides, they become USER_CONFIRMED, so they are not LOW anymore.
    # We just need to make sure minimum contract is satisfied.

    ds.recoverchain_signals = final_mappings
    ds.status = DatasetStatus.READY_FOR_ANALYSIS
    if "declared_domain" in req.model_fields_set:  # an explicit null clears a declaration
        _record_domain_declaration(
            db,
            ds,
            req.declared_domain.value if req.declared_domain else None,
            "mapping confirmation",
        )
    if req.source_system is not None:
        ds.source_system = req.source_system.strip() or None
    domain_classification = _sample_classification(service, ds)
    ds.domain_classification = (
        domain_classification.to_dict() if domain_classification is not None else None
    )
    db.commit()

    return {
        "status": "SUCCESS",
        "message": "Mapping confirmed.",
        "classification": validation_res["classification"],
        "domain_classification": ds.domain_classification,
    }


from api.schemas import (
    DatasetPredictionRequest,
    DatasetPredictionResponse,
    GenerateCasesRequest,
    GenerateCasesResponse,
)
from application.recovery_predictor_ml import MLPaymentFailurePredictor


@router.post("/{dataset_id}/predict", response_model=DatasetPredictionResponse, dependencies=_AUTH)
def shadow_predict(dataset_id: str, request: DatasetPredictionRequest):
    predictor = MLPaymentFailurePredictor(dataset_id=dataset_id, registry_dir=_registry_dir())
    if not predictor.model:
        raise HTTPException(status_code=404, detail="No valid model found for this dataset")

    try:
        res = predictor.predict_failure_risk(request.canonical_features)
        return DatasetPredictionResponse(
            probability=res.get("probability", 0.0),
            status=res.get("status", "SUCCESS"),
            model_metadata=res.get("model_metadata", {}),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


from pydantic import BaseModel


@router.post(
    "/{dataset_id}/generate-cases", response_model=GenerateCasesResponse, dependencies=_AUTH
)
def generate_cases_from_dataset(
    dataset_id: str,
    req: GenerateCasesRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    service = DatasetLabService(db)
    ds = service.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not ds.training_suitability or ds.training_suitability.get("readiness_status") not in [
        "ML_TRAINING_READY",
        "ML_TRAINING_READY_WITH_EXCLUSIONS",
        "ML_TRAINING_READY_WITH_WARNINGS",
    ]:
        raise HTTPException(status_code=400, detail="Dataset must be ML ready to generate cases")

    file_path = os.path.join(service.dataset_dir, ds.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset file missing")

    bounded_max_cases = min(max(1, req.max_cases), 500)
    import pandas as pd

    # Read one row past the cap so a truncated source is DETECTED and reported. A
    # 720-row file used to report rows_seen=500 and say nothing about the 220 rows it
    # never processed.
    _fpl = file_path.lower()
    try:
        df, parquet_rows = _read_source_frame(file_path, bounded_max_cases + 1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {e}") from e
    source_rows = parquet_rows if parquet_rows is not None else ds.row_count
    truncated = len(df) > bounded_max_cases or (
        _fpl.endswith(".parquet") and source_rows is not None and source_rows > bounded_max_cases
    )
    df = df.head(bounded_max_cases)

    # Domain classification (application/domain_classification.py), before anything
    # else runs. Explicit first - a declared dataset domain, or a column confirmed as
    # EVENT_CATEGORY - then inference from named evidence only. A dataset whose domain
    # cannot be established generates nothing: it is never assumed to be a failed
    # payment, which is what the column-name keyword heuristic used to do (a
    # subscription workbook was ingested as FAILED_PAYMENT that way).
    classification = _classify(ds, df)
    ds.domain_classification = classification.to_dict()
    if classification.needs_mapping:
        db.commit()
        raise HTTPException(
            status_code=422,
            detail={
                "code": "DOMAIN_NEEDS_MAPPING",
                "message": "The revenue domain of this dataset could not be established: "
                + classification.reason,
                "domain_classification": classification.to_dict(),
            },
        )

    # Train / refresh the per-dataset shadow model on a BACKGROUND task so it does
    # not block this response (training two calibrated candidates took ~2 s of the
    # request). This run's cases are scored by whatever model is on disk now — the
    # deterministic baseline on a first upload — and the trained model is picked up
    # on the next run and by the Insights view. Shadow-only: the model never
    # authorises an action; it only feeds the recovery-probability / ERV estimate.
    spec = ds.training_suitability or {}
    if str(spec.get("readiness_status", "")).startswith("ML_TRAINING_READY") and len(df) >= 20:
        train_spec = dict(spec)
        train_spec["dataset_id"] = dataset_id
        # Honest bar: a model that does not clear 0.55 ROC-AUC is left
        # REJECTED_LOW_QUALITY and the pipeline keeps using the baseline.
        background_tasks.add_task(
            run_ml_training_task, dataset_id, train_spec, file_path, 0.55, 200
        )

    # Extract mapping
    mappings = ds.recoverchain_signals or []
    canonical_to_original = {}
    for m in mappings:
        if m["canonical_field"] != "UNKNOWN":
            canonical_to_original[m["canonical_field"]] = m["original_column"]

    from decimal import Decimal

    from domain.models import Money, RevenueEvent

    _cur_col = next(
        (c for c in df.columns if str(c).lower() in ("currency", "ccy", "currency_code")), None
    )
    _default_ccy = os.getenv("DEFAULT_CURRENCY", "INR")

    def _json_safe(v):
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.isoformat()
        if isinstance(v, Decimal):
            return float(v)
        if hasattr(v, "item"):  # numpy scalar
            try:
                return v.item()
            except Exception:
                return str(v)
        return v

    import uuid
    from datetime import datetime

    from application.agents import AgentOrchestrator
    from application.case_engine import CaseEngine, DuplicateEventException
    from application.case_pipeline import CasePipelineService
    from application.recovery_predictor_ml import MLPaymentFailurePredictor
    from infrastructure.repositories import SqlAlchemyAuditRecorder, SqlAlchemyCaseRepository

    case_repo = SqlAlchemyCaseRepository(db)
    audit_repo = SqlAlchemyAuditRecorder(db)
    case_engine = CaseEngine(case_repo, audit_repo)
    pipeline = CasePipelineService(case_repo, audit_repo)
    agent = AgentOrchestrator()

    # Batch the writes: each accepted row used to trigger 3-4 separate commits
    # (ingest, advance, execute, verify). Flush during the loop so in-loop reads
    # still see the rows, then commit the whole batch once at the end.
    case_repo._defer_commit = True

    # Build the shadow predictor ONCE for this dataset and reuse it for every row.
    # Previously predict_recovery_for_case() re-scanned the model registry and
    # re-loaded the model on each of up to 500 rows.
    shared_predictor = MLPaymentFailurePredictor(dataset_id=dataset_id)

    from application.ingestion import (
        FILE_EXTERNAL_SYSTEM,
        IngestionRunReport,
        RejectReason,
        SourceType,
        ValidationFailure,
        derive_failure_code,
        event_fingerprint_v2,
        fingerprint_event_id_v2,
        legacy_v1_event_ids,
        normalize_currency,
        normalize_identifier,
        parse_amount,
        parse_timestamp,
    )
    from domain.lifecycle import CaseClosedError, CaseLifecycleManager
    from domain.models import CaseState, PolicyDecisionStatus
    from infrastructure.orm import EventModel

    report = IngestionRunReport(
        source_type=SourceType.FILE_UPLOAD,
        source_ref=dataset_id,
        source_name=ds.name,
        rows_in_source=source_rows,
        domain_classification=classification.to_dict(),
    )
    rows_not_processed = (
        max(0, int(source_rows) - len(df)) if truncated and source_rows is not None else 0
    )
    report.rows_not_processed = rows_not_processed

    # Every case this run resolved rows to: created, updated by correlation, or - for a
    # row that was already ingested earlier - the case that already holds it. A re-run
    # therefore still returns the cases it describes, without creating copies.
    case_ids: list[str] = []
    accepted_case_ids: set[str] = set()

    # The first eight keys keep their existing meaning (the UI and existing tests read
    # them); rows_skipped still counts every row that produced no new work. The keys
    # after them say WHY, which rows_skipped alone never did.
    counters = {
        "rows_seen": len(df),
        "rows_accepted": 0,
        "rows_skipped": 0,
        "invalid_amount": 0,
        "invalid_entity": 0,
        "invalid_timestamp": 0,
        "invalid_target": 0,
        "ambiguous_target": 0,
        "invalid_currency": 0,
        "not_at_risk": 0,
        "duplicate_event": 0,
        "case_closed": 0,
        "cases_created": 0,
        "cases_updated": 0,
        "rows_in_file": int(source_rows) if source_rows is not None else len(df),
        "rows_not_processed": rows_not_processed,
        "unresolved_domain": 0,
    }
    _COUNTER_FOR = {
        RejectReason.MISSING_AMOUNT: "invalid_amount",
        RejectReason.INVALID_AMOUNT: "invalid_amount",
        RejectReason.NON_POSITIVE_AMOUNT: "invalid_amount",
        RejectReason.AMOUNT_TOO_LARGE: "invalid_amount",
        RejectReason.INVALID_CURRENCY: "invalid_currency",
        RejectReason.MISSING_IDENTIFIER: "invalid_entity",
        RejectReason.INVALID_TIMESTAMP: "invalid_timestamp",
        RejectReason.MISSING_TARGET: "invalid_target",
        RejectReason.AMBIGUOUS_TARGET: "ambiguous_target",
    }
    _NOT_FAILED = {"0", "false", "no", "success", "paid", "settled", "completed"}
    _FAILED = {
        "1",
        "true",
        "yes",
        "failed",
        "unpaid",
        "returned",
        "declined",
        "error",
        "insufficient_funds",
    }

    amount_col = canonical_to_original.get("AMOUNT") or canonical_to_original.get("BALANCE")
    target_col = canonical_to_original.get("TARGET") or canonical_to_original.get("OUTCOME")
    entity_col = (
        canonical_to_original.get("ENTITY_ID")
        or canonical_to_original.get("CUSTOMER_ID")
        or canonical_to_original.get("ACCOUNT_ID")
    )
    time_col = canonical_to_original.get("TIMESTAMP")
    tx_col = canonical_to_original.get("TRANSACTION_ID")
    occurrences: dict[str, int] = {}

    try:
        for idx, row in df.iterrows():
            report.rows_received += 1
            row_no = int(idx) + 1  # 1-based data row, header excluded
            row_dict = row.to_dict()

            # Validation - the same contract the /events API applies
            # (application/ingestion.py). Checks run in the order they always have.
            try:
                if not amount_col:
                    raise ValidationFailure(
                        RejectReason.MISSING_AMOUNT, "amount", "no column is mapped to AMOUNT"
                    )
                amt = parse_amount(row_dict.get(amount_col), amount_col)

                if not target_col:
                    raise ValidationFailure(
                        RejectReason.MISSING_TARGET, "outcome", "no column is mapped to OUTCOME"
                    )
                outcome_val = row_dict.get(target_col)
                if pd.isna(outcome_val):
                    raise ValidationFailure(RejectReason.MISSING_TARGET, target_col, "is empty")
                outcome_str = str(outcome_val).strip().lower()
                if outcome_str in _NOT_FAILED:
                    is_failed = False
                elif outcome_str in _FAILED:
                    is_failed = True
                else:
                    raise ValidationFailure(
                        RejectReason.AMBIGUOUS_TARGET,
                        target_col,
                        f"{outcome_val!r} is neither a failure nor a success",
                    )

                if is_failed:
                    if not entity_col:
                        raise ValidationFailure(
                            RejectReason.MISSING_IDENTIFIER,
                            "customer",
                            "no column is mapped to a customer or account id",
                        )
                    customer_id = normalize_identifier(row_dict.get(entity_col), entity_col)
                    if not time_col:
                        raise ValidationFailure(
                            RejectReason.INVALID_TIMESTAMP,
                            "timestamp",
                            "no column is mapped to TIMESTAMP",
                        )
                    ts = parse_timestamp(row_dict.get(time_col), time_col)
                    ccy = normalize_currency(
                        row_dict.get(_cur_col) if _cur_col else None, default=_default_ccy
                    )
            except ValidationFailure as vf:
                counters["rows_skipped"] += 1
                counters[_COUNTER_FOR[vf.reason]] += 1
                report.reject(vf.reason, row=row_no, field_name=vf.field_name, detail=vf.detail)
                continue

            if not is_failed:
                counters["rows_skipped"] += 1
                counters["not_at_risk"] += 1
                report.rows_not_at_risk += 1
                continue

            # The row's domain: the dataset's, or - per row - its category column value,
            # which must name a known domain. An unresolved row is rejected, never
            # defaulted to a failed payment.
            resolved = classification.resolve(row_dict)
            if resolved is None:
                counters["rows_skipped"] += 1
                counters["unresolved_domain"] += 1
                report.reject(
                    RejectReason.UNRESOLVED_DOMAIN,
                    row=row_no,
                    field_name=classification.category_column,
                    detail=(
                        f"{row_dict.get(classification.category_column)!r} does not name a "
                        "known revenue domain"
                    ),
                )
                continue
            risk_category, domain_basis = resolved

            # Only a REAL transaction id may link rows to one obligation. When the file
            # has none the obligation cannot be identified, so nothing is correlated -
            # the rule CaseEngine documents. A positional "tx_<row index>" used to be
            # invented here, so row N of one file correlated with row N of any other
            # file for the same customer and category, and overwrote its amount.
            reference_id = None
            if tx_col:
                raw_ref = row_dict.get(tx_col)
                if raw_ref is not None and not pd.isna(raw_ref) and str(raw_ref).strip():
                    reference_id = str(raw_ref).strip()

            # Surface a failure reason for the diagnosis engine: any column whose name
            # looks like a reason/cause/decline code becomes raw_payload["failure_code"].
            # M1c - every value is made JSON-safe (Timestamp / Decimal / numpy) so the
            # raw_payload JSON column never fails to serialise.
            payload = {k: _json_safe(v) for k, v in row_dict.items()}
            failure_code = derive_failure_code(payload)
            if failure_code is not None and payload.get("failure_code") is None:
                payload["failure_code"] = failure_code

            # Event identity comes from the row's CONTENT, not from this upload: the same
            # failure uploaded again - in another file or another run - is the same event.
            # v2 identity leaves the domain out, so a dataset whose domain is declared
            # after an earlier run inferred another one still recognises its rows.
            fingerprint = event_fingerprint_v2(
                customer_id=customer_id,
                amount=amt,
                currency=ccy,
                timestamp=ts,
                reference_id=reference_id,
                failure_code=failure_code,
            )
            occurrence = occurrences.get(fingerprint, 0)
            occurrences[fingerprint] = occurrence + 1
            external_event_id = fingerprint_event_id_v2(fingerprint, occurrence)

            # Already stored - under this identity, under the Phase 3 v1 identity (which
            # hashed in the category), or under an alias written by identity
            # reconciliation for pre-Phase-3 events?
            existing = case_repo.find_existing_event(
                FILE_EXTERNAL_SYSTEM,
                [
                    external_event_id,
                    *legacy_v1_event_ids(
                        customer_id=customer_id,
                        amount=amt,
                        currency=ccy,
                        timestamp=ts,
                        reference_id=reference_id,
                        failure_code=failure_code,
                        occurrence=occurrence,
                    ),
                ],
            )
            if existing is not None:
                counters["rows_skipped"] += 1
                counters["duplicate_event"] += 1
                report.rows_duplicate += 1
                audit_repo.log_transition(
                    case_id="SYSTEM",
                    from_state="N/A",
                    to_state="N/A",
                    evidence={
                        "action": "deduplicated",
                        "external_id": external_event_id,
                        "matched_event_id": existing.event_id,
                        "matched_external_event_id": existing.external_event_id,
                        "matched_case_id": existing.case_id,
                        "ingestion_run_id": report.run_id,
                        "row": row_no,
                    },
                )
                if existing.case_id and existing.case_id not in case_ids:
                    case_ids.append(existing.case_id)
                continue

            event = RevenueEvent(
                event_id=f"evt_{uuid.uuid4().hex[:8]}",
                customer_id=customer_id,
                risk_category=risk_category,
                external_system=FILE_EXTERNAL_SYSTEM,
                external_event_id=external_event_id,
                reference_id=reference_id,
                amount=Money(amount=amt, currency=ccy),
                timestamp=ts,
                raw_payload=payload,
                source_type=SourceType.FILE_UPLOAD,
                source_ref=dataset_id,
                ingestion_run_id=report.run_id,
                domain_basis=domain_basis,
            )

            try:
                case, is_new = case_engine.ingest_normalized_event(event)
            except DuplicateEventException:
                counters["rows_skipped"] += 1
                counters["duplicate_event"] += 1
                report.rows_duplicate += 1
                existing_case_id = (
                    db.query(EventModel.case_id)
                    .filter(
                        EventModel.external_system == event.external_system,
                        EventModel.external_event_id == event.external_event_id,
                    )
                    .scalar()
                )
                if existing_case_id and existing_case_id not in case_ids:
                    case_ids.append(existing_case_id)
                continue

            if is_new:
                counters["cases_created"] += 1
                report.cases_created += 1
            else:
                counters["cases_updated"] += 1
                report.cases_updated += 1

            # Run the same deterministic pipeline the live case flow uses:
            # risk -> diagnose -> predict (shadow ML or baseline) -> recommend (+ERV)
            # -> policy. Then, if policy permitted, act and verify in the sandbox.
            #
            # A row can correlate onto an EXISTING case that has already dispatched an
            # action (same customer + category + reference_id). Reprocessing it would
            # rewind that case, so it is skipped instead - and skipping must not take the
            # whole batch down with it.
            try:
                case = pipeline.advance(case, dataset_id=dataset_id, predictor=shared_predictor)
            except CaseClosedError:
                counters["rows_skipped"] += 1
                counters["case_closed"] += 1
                report.reject(
                    RejectReason.CASE_CLOSED,
                    row=row_no,
                    detail=f"correlated case {case.case_id} is already closed",
                )
                continue

            if (
                case.policy_decision
                and case.policy_decision.status == PolicyDecisionStatus.PERMITTED
                and case.recommendation
                and case.recommendation.top_candidate
            ):
                record = agent.execute(
                    case, case.recommendation.top_candidate.action_type, repo=case_repo
                )
                try:
                    CaseLifecycleManager.record_execution_result(
                        case,
                        CaseState.PENDING_VERIFICATION
                        if record.status.value == "COMPLETED_SIMULATED"
                        else case.current_state,
                    )
                except CaseClosedError:
                    pass  # case closed since correlation (shouldn't happen - see case_engine
                    # ACTIVE_STATES); leave it untouched rather than overwrite a closed case.
                else:
                    case.execution_record = record
                    case_repo.save(case)

            # Phase 5: execution is where this pipeline stops. A row replayed from a
            # file can simulate an ACTION (MockExecutionAdapter, "COMPLETED_SIMULATED")
            # but never a real settlement - ACTION ATTEMPT != REVENUE RECOVERED. The
            # case is left in PENDING_VERIFICATION (or wherever execution put it) and
            # only moves to a recovered/cancelled state when an authoritative outcome
            # event is ingested for the same obligation (POST /outcomes /
            # /outcomes/batch, application/outcome_events.py). File-based outcome rows
            # are not part of this phase (see the Phase 5 report's dataset
            # specification); a settlement dataset is future work.

            report.rows_accepted += 1
            report.accepted_by_domain[risk_category.value] = (
                report.accepted_by_domain.get(risk_category.value, 0) + 1
            )
            if case.case_id not in case_ids:
                case_ids.append(case.case_id)
            if case.case_id not in accepted_case_ids:
                accepted_case_ids.add(case.case_id)
                counters["rows_accepted"] += 1

        # Commit the whole batch once, together with the record of what happened to it.
        case_repo._defer_commit = False
        db.add(report.to_model("COMPLETED"))
        db.commit()
    except Exception as exc:
        # The run shares one transaction, so nothing from a failed run is kept - but the
        # failure itself is recorded instead of disappearing with the response.
        case_repo._defer_commit = False
        db.rollback()
        db.add(
            report.to_model(
                "FAILED",
                error_message=f"rolled back - {type(exc).__name__}: {exc}"[:500],
            )
        )
        db.commit()
        raise

    return {
        "status": "SUCCESS",
        "cases_generated": len(case_ids),
        "case_ids": case_ids,
        "counters": counters,
        "ingestion_run_id": report.run_id,
        "truncated": truncated,
        "rejected_samples": report.rejected_samples,
        "domain_classification": classification.to_dict(),
    }


@router.get("/{dataset_id}/ingestion-runs")
def list_dataset_ingestion_runs(dataset_id: str, limit: int = 20, db: Session = Depends(get_db)):
    """Case-generation runs for this dataset, newest first, each with its data-quality
    result: rows accepted, rejected, duplicate, not at risk and not processed, and why."""
    from application.ingestion import ingestion_run_to_dict
    from infrastructure.orm import IngestionRunModel

    limit = min(max(1, limit), 100)
    rows = (
        db.query(IngestionRunModel)
        .filter(
            IngestionRunModel.source_type == "FILE_UPLOAD",
            IngestionRunModel.source_ref == dataset_id,
        )
        .order_by(IngestionRunModel.started_at.desc())
        .limit(limit)
        .all()
    )
    return [ingestion_run_to_dict(r) for r in rows]
