from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_question_validation():
    response = client.post(
        "/v1/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


@patch("main.requests.post")
def test_ask_question(mock_post):
    fake_response = Mock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "message": {
            "content": "Une API REST permet à des applications de communiquer."
        }
    }

    mock_post.return_value = fake_response

    response = client.post(
        "/v1/ask",
        json={
            "question": "Qu'est-ce qu'une API REST ?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert data["answer"] == (
        "Une API REST permet à des applications de communiquer."
    )
    assert data["sources"] == []
    assert "request_id" in data