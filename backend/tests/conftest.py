import os
import time

# ---------------------------------------------------------------------------
# Test database isolation.
# Force every test run onto a dedicated throwaway SQLite file so integration
# tests never share state with `run.py` (which uses ./test_recoverchain.db) or
# with a developer's real database. This MUST run before any test module
# imports `infrastructure.db`, so it lives at import time in conftest.py.
# Override by exporting RECOVERCHAIN_TEST_DATABASE_URL.
# ---------------------------------------------------------------------------
_TEST_DB_FILE = os.path.join(os.path.dirname(__file__), os.pardir, "_pytest_recoverchain.db")
_TEST_DB_URL = os.getenv(
    "RECOVERCHAIN_TEST_DATABASE_URL",
    "sqlite:///" + os.path.abspath(_TEST_DB_FILE).replace("\\", "/"),
)
os.environ["DATABASE_URL"] = _TEST_DB_URL

# Test storage isolation. Uploaded dataset files and trained model artefacts go to a
# throwaway folder as well. The suite used to share the development app's ../dataset
# and ml/models/registry folders, and its dataset sync, upload pruning and orphan-model
# sweep deleted the development app's files. Always overridden (never setdefault), and
# removed at the end of the session.
import shutil  # noqa: E402
import tempfile  # noqa: E402

_TEST_STORAGE_ROOT = tempfile.mkdtemp(prefix="recoverchain_test_storage_")
os.environ["RECOVERCHAIN_DATASET_DIR"] = os.path.join(_TEST_STORAGE_ROOT, "dataset")
os.environ["RECOVERCHAIN_MODEL_REGISTRY_DIR"] = os.path.join(_TEST_STORAGE_ROOT, "registry")
os.makedirs(os.environ["RECOVERCHAIN_DATASET_DIR"], exist_ok=True)
os.makedirs(os.environ["RECOVERCHAIN_MODEL_REGISTRY_DIR"], exist_ok=True)

# The ML quality gate (ml_training) would reject the deliberately tiny fixtures
# that plumbing tests train. Disable it by default for the suite; the dedicated
# gate test re-enables it explicitly.
os.environ.setdefault("ML_MIN_ROC_AUC", "0.0")
os.environ.setdefault("ML_MIN_TEST_ROWS", "0")

# The in-process rate limiter would trip on the hundreds of requests a full
# suite run fires from a single TestClient identity. A dedicated test exercises
# it explicitly by re-enabling this.
os.environ.setdefault("RATE_LIMIT_ENABLED", "0")

if _TEST_DB_URL.startswith("sqlite:///") and os.path.exists(os.path.abspath(_TEST_DB_FILE)):
    try:
        os.remove(os.path.abspath(_TEST_DB_FILE))
    except OSError:
        pass

import pytest


@pytest.fixture(scope="session", autouse=True)
def _test_database():
    """Create a clean schema for the session and drop the file afterwards."""
    import infrastructure.dataset_orm  # noqa: F401
    import infrastructure.orm  # noqa: F401  (register models)
    from infrastructure.db import Base, engine

    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    shutil.rmtree(_TEST_STORAGE_ROOT, ignore_errors=True)
    if _TEST_DB_URL.startswith("sqlite:///"):
        try:
            os.remove(os.path.abspath(_TEST_DB_FILE))
        except OSError:
            pass


@pytest.fixture(scope="session", autouse=True)
def _bypass_api_key(_test_database):
    """Most tests don't care about auth. Override the API-key dependency for the
    whole suite; test_auth.py clears this to exercise the 401/403 path."""
    from api.access import (
        require_admin,
        require_operator,
        require_principal,
        require_reviewer,
    )
    from api.main import app, verify_api_key
    from api.principals import Principal

    admin = Principal("test-principal", "ADMIN", "test-bypass")
    reviewer = Principal("test-reviewer", "REVIEWER", "test-bypass")
    overrides = {
        verify_api_key: lambda: "test-bypass",
        require_principal: lambda: admin,
        require_operator: lambda: admin,
        require_reviewer: lambda: reviewer,
        require_admin: lambda: admin,
    }
    app.dependency_overrides.update(overrides)
    yield
    for dep in overrides:
        app.dependency_overrides.pop(dep, None)


@pytest.fixture(autouse=True)
def remove_artificial_delays(monkeypatch):
    """
    Optimizes test execution by stripping artificial simulated delays
    (e.g., from SimulatedLLMAdapter or other simulated infra).
    """

    # Patch time.sleep globally during tests to eliminate artificial latency
    def mock_sleep(seconds):
        pass

    monkeypatch.setattr(time, "sleep", mock_sleep)


def pytest_collection_modifyitems(config, items):
    for item in items:
        # Determine if test is integration or fast
        is_integration = False

        # Check by file name
        if item.fspath.basename in [
            "test_api_endpoints.py",
            "test_dataset_lab.py",
            "test_llm.py",
            "test_evaluation.py",
            "test_dataset_api.py",
        ]:
            is_integration = True

        # Check by specific test name (e.g. hits API)
        if "api" in item.name and "endpoint" in item.name:
            is_integration = True

        # Check by fixture requests
        if hasattr(item, "fixturenames"):
            if (
                "client" in item.fixturenames
                or "session" in item.fixturenames
                or "db" in item.fixturenames
            ):
                is_integration = True

        if is_integration:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.fast)
