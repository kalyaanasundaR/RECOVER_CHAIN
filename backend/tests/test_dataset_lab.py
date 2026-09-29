import os
import uuid

from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.main import app
from infrastructure.db import engine, get_db
from infrastructure.storage import dataset_dir

client = TestClient(app)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


def test_sync_datasets():
    os.makedirs(dataset_dir(), exist_ok=True)
    fname = f"test_dummy_{uuid.uuid4().hex}.csv"
    with open(os.path.join(dataset_dir(), fname), "w") as f:
        f.write("a,b,c\n1,2,3")

    res = client.post("/datasets/sync")
    assert res.status_code == 200
    data = res.json()
    assert "imported_count" in data

    # Find our specific file id
    ds_id = None
    all_res = client.get("/datasets").json()["datasets"]
    for d in all_res:
        if d["filename"] == fname:
            ds_id = d["dataset_id"]
            break

    assert ds_id is not None

    res2 = client.post(f"/datasets/{ds_id}/analyze")
    assert res2.status_code == 200

    res3 = client.get(f"/datasets/{ds_id}")
    assert res3.status_code == 200
    assert res3.json()["row_count"] == 1


def test_dataset_upload():
    file_content = b"customer_id,amount,status\n1,100,success"
    fname = f"upload_test_{uuid.uuid4().hex}.csv"
    files = {"file": (fname, file_content, "text/csv")}
    res = client.post("/datasets/upload", files=files)
    assert res.status_code == 200
    ds_id = res.json()["dataset_id"]

    res_analyze = client.post(f"/datasets/{ds_id}/analyze")
    assert res_analyze.status_code == 200

    res_get = client.get(f"/datasets/{ds_id}")
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["name"] == fname
    assert data["row_count"] == 1


def test_security_extension():
    file_content = b"print('hack')"
    files = {"file": ("upload_test.py", file_content, "text/x-python")}
    res = client.post("/datasets/upload", files=files)
    assert res.status_code == 400
    assert "Unsupported file format" in res.json()["detail"]
