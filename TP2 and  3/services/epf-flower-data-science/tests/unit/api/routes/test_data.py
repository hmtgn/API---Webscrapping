import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys
import os
sys.path.append(r"C:\Users\hugo.montagnon\Desktop\Divers\EPF\Data Source\API_Project\TP2 and  3\services\epf-flower-data-science")
from main import app  

client = TestClient(app)



# Test dataset download endpoint
def test_download_dataset():
    response = client.get("/api/download-dataset/iris")
    assert response.status_code == 200
    assert "path" in response.json()

# Test dataset addition
def test_manage_datasets_add():
    response = client.post(
        "/api/manage-datasets",
        data={"dataset_name": "New Dataset", "dataset_url": "new-dataset-url"}
    )
    assert response.status_code == 200
    assert "New Dataset" in response.text

# Test dataset listing
def test_show_datasets():
    response = client.get("/api/datasets")
    assert response.status_code == 200
    assert "<table" in response.text

# Test model training
def test_train_model():
    response = client.post("/api/train-model")
    assert response.status_code == 200
    assert "model_path" in response.json()

# Test prediction
def test_prediction():
    response = client.post(
        "/api/predict",
        json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()

# Test protected endpoint with rate limiting
def test_protected_endpoint_rate_limit():
    for _ in range(5):
        response = client.get("/api/protected-endpoint")
        assert response.status_code == 200

    # Sixth request should be rate-limited
    response = client.get("/api/protected-endpoint")
    assert response.status_code == 429
    assert "rate limit exceeded" in response.text.lower()
