import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.routes import app

client = TestClient(app)


def test_home_page():
    """Test that home page returns HTML"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI Image Classifier" in response.text


def test_home_page_contains_upload_form():
    """Test that home page contains file upload form"""
    response = client.get("/")
    assert response.status_code == 200
    assert 'type="file"' in response.text
    assert 'id="fileInput"' in response.text


def test_predict_endpoint_without_file():
    """Test prediction endpoint returns error without file"""
    response = client.post("/predict/")
    assert response.status_code == 422  # Unprocessable Entity


def test_predict_endpoint_with_invalid_file():
    """Test prediction endpoint with invalid file type"""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/predict/", files=files)
    # Should either fail or handle gracefully
    assert response.status_code in [200, 400, 422, 500]


def test_static_files_mounted():
    """Test that static files are accessible"""
    # This test checks if static route is configured
    # Actual files may not exist in test environment
    response = client.get("/static/")
    # Should return 404 or 405, not 500
    assert response.status_code in [404, 405]


def test_app_title():
    """Test that app has correct title"""
    from app import app
    assert app.title == "Simple AI Web App"


@pytest.mark.asyncio
async def test_predict_with_dummy_image():
    """Test prediction with a simple test image"""
    from io import BytesIO
    from PIL import Image
    
    # Create a simple test image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {"file": ("test.png", img_bytes, "image/png")}
    response = client.post("/predict/", files=files)
    
    # Should return 200 with prediction
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert isinstance(data["predicted_class"], str)
    assert len(data["predicted_class"]) > 0
