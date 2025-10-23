from .model import load_server_history, save_server_history, predict_image, load_model
from PIL import Image
import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import os

app = FastAPI(title="Simple AI Web App")
app.mount("/static", StaticFiles(directory="static"), name="static")

model = load_model()

@app.get("/", response_class=HTMLResponse)
def home():
    this_dir = os.path.dirname(__file__)
    with open(os.path.join(this_dir, "interface.html")) as f:
        return f.read()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and return top-k predictions with confidence scores."""
    image = Image.open(file.file).convert("RGB")
    top_predictions = predict_image(image, model)
    return JSONResponse({"predicted_class": top_predictions[0]["label"], "predictions": top_predictions})


@app.get("/history")
async def get_history():
    """Return server-side stored prediction history."""
    items = load_server_history()
    return JSONResponse({"history": items})


@app.post("/history")
async def post_history(request: Request):
    """Save a prediction entry to server-side history.

    Expects JSON: { label: str, dataUrl: str (optional), t: int }
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not isinstance(payload, dict) or "label" not in payload:
        return JSONResponse({"error": "Missing label"}, status_code=400)

    items = load_server_history()
    items.append(payload)
    # keep only recent 200 entries to avoid unbounded growth
    if len(items) > 200:
        items = items[-200:]
    save_server_history(items)
    return JSONResponse({"ok": True})


@app.delete("/history")
async def delete_history():
    """Clear server-side history file."""
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"ok": True})


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Handle image upload and return prediction."""
    if file:
        # Process the uploaded file
        contents = await file.read()
        # Add your processing logic here
        return JSONResponse(content={"message": "File uploaded successfully!"})
    return JSONResponse(content={"error": "No file uploaded"}, status_code=400)
