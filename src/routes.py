from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .model import load_model, predict_image, labels
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
    image = Image.open(file.file).convert("RGB")
    predicted_label = predict_image(image, model, labels)
    return JSONResponse({"predicted_class": predicted_label})
