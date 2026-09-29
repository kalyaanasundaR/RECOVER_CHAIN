import enum

from sqlalchemy import JSON, Column, DateTime, Enum, Integer, String

from infrastructure.db import Base


class DatasetStatus(str, enum.Enum):
    PENDING = "PENDING"
    UPLOADED = "UPLOADED"
    PROFILING = "PROFILING"
    MAPPING_REVIEW = "MAPPING_REVIEW"
    READY_FOR_ANALYSIS = "READY_FOR_ANALYSIS"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    ML_READY = "ML_READY"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    FAILED = "FAILED"


class DatasetMetadataModel(Base):
    __tablename__ = "datasets"

    dataset_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    # SHA-256 of the uploaded bytes (Phase 3): identifies a byte-identical re-upload.
    content_sha256 = Column(String, index=True, nullable=True)
    upload_timestamp = Column(DateTime, nullable=False)
    status = Column(Enum(DatasetStatus), nullable=False, default=DatasetStatus.PENDING)
    error_message = Column(String, nullable=True)

    # General Stats
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    duplicate_row_count = Column(Integer, nullable=True)
    total_missing_values = Column(Integer, nullable=True)

    # Detailed Analysis (Stored as JSON for flexibility)
    columns_profile = Column(JSON, nullable=True)
    data_quality_report = Column(JSON, nullable=True)
    temporal_analysis = Column(JSON, nullable=True)
    training_suitability = Column(JSON, nullable=True)
    recoverchain_signals = Column(JSON, nullable=True)
    target_detection = Column(JSON, nullable=True)
    leakage_detection = Column(JSON, nullable=True)

    # Dataset category contract (Phase 4). A declared domain is used as given; without
    # one, generate-cases infers only from deterministic evidence and otherwise refuses
    # (application/domain_classification.py). domain_classification holds the last
    # classification with its evidence. Schema = recoverchain_signals, version =
    # content_sha256, runs = ingestion_runs.
    declared_domain = Column(String, nullable=True)  # a RiskCategory value
    source_system = Column(String, nullable=True)  # e.g. "stripe", "netsuite"; provenance only
    domain_classification = Column(JSON, nullable=True)
