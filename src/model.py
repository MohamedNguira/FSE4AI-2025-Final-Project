from torchvision import models, transforms
import torch
import urllib.request
import os

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

def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model

def predict_image(image, model, labels):
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        predicted_label = labels[predicted.item()]
    return predicted_label
