import os
import uuid
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from infrastructure.dataset_orm import DatasetMetadataModel, DatasetStatus
from infrastructure.storage import dataset_dir as configured_dataset_dir


class DatasetLabService:
    def __init__(self, db: Session, dataset_dir: str | None = None):
        self.db = db
        # The configured upload folder (infrastructure/storage.py) unless one is given,
        # so tests and deployments never share the development app's files.
        self.dataset_dir = dataset_dir or configured_dataset_dir()
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

    def import_local_datasets(self):
        """Scans the dataset directory and adds new files to the database."""
        imported = []
        for filename in os.listdir(self.dataset_dir):
            if filename.startswith("."):
                continue

            file_path = os.path.join(self.dataset_dir, filename)
            if os.path.isfile(file_path):
                ext = filename.lower().split(".")[-1]
                if "zip" in filename.lower():
                    ext = "zip"  # We will attempt to parse .csv.zip

                # Check if already exists in DB
                existing = (
                    self.db.query(DatasetMetadataModel)
                    .filter(DatasetMetadataModel.filename == filename)
                    .first()
                )
                if not existing:
                    size = os.path.getsize(file_path)
                    ds_id = f"ds_{uuid.uuid4().hex[:8]}"
                    new_ds = DatasetMetadataModel(
                        dataset_id=ds_id,
                        name=filename,
                        filename=filename,
                        file_type=ext,
                        file_size_bytes=size,
                        upload_timestamp=datetime.now(UTC),
                        status=DatasetStatus.UPLOADED,
                    )
                    self.db.add(new_ds)
                    imported.append(ds_id)

        self.db.commit()
        return imported

    def get_all_datasets(self):
        return (
            self.db.query(DatasetMetadataModel)
            .order_by(DatasetMetadataModel.upload_timestamp.desc())
            .all()
        )

    def get_dataset(self, dataset_id: str):
        return (
            self.db.query(DatasetMetadataModel)
            .filter(DatasetMetadataModel.dataset_id == dataset_id)
            .first()
        )

    def analyze_dataset(self, dataset_id: str):
        ds = self.get_dataset(dataset_id)
        if not ds:
            raise ValueError("Dataset not found")

        ds.status = DatasetStatus.PROFILING
        self.db.commit()

        file_path = os.path.join(self.dataset_dir, ds.filename)
        try:
            from application.dataset_intelligence import DatasetProfiler

            # Use chunked/sampled profiling logic
            profile = DatasetProfiler.profile_file(file_path, ds.file_type)

            ds.row_count = profile["row_count"]
            ds.column_count = profile["column_count"]
            ds.columns_profile = profile["columns_profile"]

            # Map semantic fields to signals format expected by the rest of the application
            ds.recoverchain_signals = profile["mapped_schema"]
            ds.leakage_detection = profile["leakage_warnings"]
            ds.data_quality_report = profile.get("data_quality_score") or {}

            validation = profile["validation"]
            ds.training_suitability = {
                "overall_classification": validation["classification"],
                "reason": validation["reason"],
            }

            ds.status = DatasetStatus.MAPPING_REVIEW

        except Exception as e:
            ds.status = DatasetStatus.FAILED
            ds.error_message = str(e)

        self.db.commit()
        return ds
