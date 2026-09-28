from fastapi.testclient import TestClient

from autoinsight_ai.api.main import app


def test_health_reports_pipeline_readiness_separately():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["pipeline_status"] == "not_implemented"
