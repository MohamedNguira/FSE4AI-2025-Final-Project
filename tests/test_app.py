import pytest
from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.routes import app


@pytest.fixture
def client():
    """Create a test client fixture"""
    return TestClient(app)


def test_home_page(client):
    """Test that home page returns HTML"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_endpoint_without_file(client):
    """Test prediction endpoint returns error without file"""
    response = client.post("/predict/")
    assert response.status_code == 422  # Unprocessable Entity


def test_get_history(client):
    """Test getting history"""
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data


def test_app_title():
    """Test that app has correct title"""
    from src.routes import app
    assert app.title == "Simple AI Web App"


@pytest.mark.asyncio
async def test_predict_with_dummy_image(client):
    """Test prediction with a simple test image"""
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