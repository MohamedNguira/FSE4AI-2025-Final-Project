from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from torchvision import models, transforms
from PIL import Image
import torch
import urllib.request

app = FastAPI(title="Simple AI Web App")

# Mount static folder (optional, for images or JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load pretrained lightweight model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# Load labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(LABELS_URL, "imagenet_classes.txt")
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]


@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the simple HTML interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Image Classifier</title>
        <style>
            body { 
                font-family: sans-serif; 
                display: flex; 
                flex-direction: column; 
                align-items: center; 
                justify-content: center; 
                min-height: 100vh; 
                background: #f8f9fa;
            }
            #preview {
                width: 224px;
                height: 224px;
                object-fit: cover;
                border: 2px solid #ccc;
                margin-top: 10px;
                border-radius: 12px;
            }
            button {
                margin-top: 15px;
                padding: 10px 20px;
                border: none;
                background: #007bff;
                color: white;
                font-size: 16px;
                border-radius: 8px;
                cursor: pointer;
            }
            button:hover {
                background: #0056b3;
            }
            #result {
                margin-top: 20px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h2>Upload an Image for Classification</h2>
        <input type="file" id="fileInput" accept="image/*">
        <img id="preview" src="#" alt="Image preview" style="display:none;">
        <button onclick="sendImage()">Predict</button>
        <div id="result"></div>

        <script>
            const fileInput = document.getElementById('fileInput');
            const preview = document.getElementById('preview');
            const result = document.getElementById('result');

            fileInput.addEventListener('change', () => {
                const file = fileInput.files[0];
                if (file) {
                    preview.src = URL.createObjectURL(file);
                    preview.style.display = 'block';
                }
            });

            async function sendImage() {
                const file = fileInput.files[0];
                if (!file) {
                    alert("Please select an image first!");
                    return;
                }

                const formData = new FormData();
                formData.append("file", file);

                result.innerText = "Predicting...";

                const response = await fetch("/predict/", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();
                result.innerText = "Predicted class: " + data.predicted_class;
            }
        </script>
    </body>
    </html>
    """


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and return prediction."""
    image = Image.open(file.file).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        predicted_label = labels[predicted.item()]

    return JSONResponse({"predicted_class": predicted_label})
