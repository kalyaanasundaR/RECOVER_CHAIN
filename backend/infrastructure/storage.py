"""Filesystem locations for uploaded dataset files and trained model artefacts.

Both default to the locations RecoverChain has always used, relative to the backend
working directory. They are configurable so that something other than the development
app - the test suite in particular - never shares them: the suite used to write into,
sync from and prune the development app's folders, and deleted its uploaded files and
trained models (observed in Phase 4). tests/conftest.py points both at a throwaway
folder before the application is imported.
"""

import os

DATASET_DIR_ENV = "RECOVERCHAIN_DATASET_DIR"
MODEL_REGISTRY_DIR_ENV = "RECOVERCHAIN_MODEL_REGISTRY_DIR"

DEFAULT_DATASET_DIR = "../dataset"
DEFAULT_MODEL_REGISTRY_DIR = "ml/models/registry"


def dataset_dir() -> str:
    """Folder holding uploaded dataset files (read at call time, not import time)."""
    return os.getenv(DATASET_DIR_ENV) or DEFAULT_DATASET_DIR


def model_registry_dir() -> str:
    """Folder holding trained model artefacts and their metadata files."""
    return os.getenv(MODEL_REGISTRY_DIR_ENV) or DEFAULT_MODEL_REGISTRY_DIR
