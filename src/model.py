import torch
import urllib.request
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from torchvision import models, transforms
import os
import json
from typing import List

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS_PATH = "imagenet_classes.txt"

# Download labels if not present
if not os.path.exists(LABELS_PATH):
    urllib.request.urlretrieve(LABELS_URL, LABELS_PATH)

with open(LABELS_PATH) as f:
    labels = [line.strip() for line in f.readlines()]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])


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

# Server-side history file
HISTORY_FILE = "predictions.json"

def load_server_history() -> List[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return data
    except Exception:
        return []
    return []

def save_server_history(items: List[dict]):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as fh:
            json.dump(items, fh)
    except Exception as e:
        print("Failed to save history:", e)

def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model

def predict_image(image, model):
    img_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        topk = probs.topk(3)
        confidences = topk.values[0].cpu().tolist()
        indices = topk.indices[0].cpu().tolist()

    top_predictions = []
    for idx, score in zip(indices, confidences):
        top_predictions.append({
            "label": labels[idx],
            "score": float(score)
        })
    return top_predictions
