from fastapi.testclient import TestClient

from server import app

client = TestClient(app)


def test_predict():
    request_response = client.post("/model", json={"input_text": "if"})
    assert request_response.status_code == 200
    assert request_response.json()["generated_text"] is not None
