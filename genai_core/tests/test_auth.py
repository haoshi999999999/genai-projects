import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_invalid_password(client):
    payload = {
        "email": "bob@example.com",
        "password": "wrong_password_123"
    }

    response = client.post(
        "/login",
        json=payload
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid credentials"


def test_login_missing_password(client):
    payload = {
        "email": "bob@example.com"
    }

    response = client.post(
        "/login",
        json=payload
    )

    assert response.status_code == 422


def test_unknown_user(client):
    payload = {
        "email": "unknown_user@example.com",
        "password": "some_password_123"
    }

    response = client.post(
        "/login",
        json=payload
    )

    assert response.status_code == 401

def test_login_success(client):
    payload = {
        "email": "bob@test.com",
        "password": "Bob@123456"
    }

    response = client.post("/login", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "logged_in"
    assert data["token_type"] == "bearer"
    assert "access_token" in data
    assert len(data["access_token"]) > 0


def test_protected_endpoint_with_valid_token(client):

    # Step 1: Login and obtain JWT
    login_payload = {
        "email": "bob@test.com",
        "password": "Bob@123456"
    }

    login_response = client.post(
        "/login",
        json=login_payload
    )

    assert login_response.status_code == 200

    token = login_response.json()["access_token"]

    # Step 2: Send JWT to protected endpoint
    headers = {
        "Authorization": f"Bearer {token}"
    }

    response = client.get(
        "/session/list",
        headers=headers
    )

    # Step 3: Verify successful authentication
    assert response.status_code == 200

    # Step 4: Verify response structure
    assert isinstance(response.json(), dict)


def test_protected_endpoint_without_token(client):

    response = client.get("/session/list")

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required"


def test_protected_endpoint_with_invalid_token(client):

    headers = {
        "Authorization": "Bearer invalid_token_123"
    }

    response = client.get(
        "/session/list",
        headers=headers
    )

    assert response.status_code == 401

    assert response.json()["detail"] == "Invalid or expired token"