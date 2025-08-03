import pytest
from fastapi.testclient import TestClient
from shelf_analyzer.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data


def test_analyze_no_file():
    """Test analyze endpoint without file"""
    response = client.post("/analyze")
    assert response.status_code == 422  # Validation error


# Add more tests as needed
