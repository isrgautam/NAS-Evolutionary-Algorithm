import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import json
import torchvision.transforms as transforms

from search_space import SuperNet

# ---------------------------
# Load architecture
# ---------------------------
with open("best_architecture.json", "r") as f:
    BEST_ARCH = json.load(f)

# ---------------------------
# Load CLASS NAMES from labels.txt
# ---------------------------
with open("labels.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

print("Loaded classes:", CLASS_NAMES)
print("Number of classes =", len(CLASS_NAMES))

# ---------------------------
# Load MODEL
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SuperNet(num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load("final_model_best_weights.pth", map_location=DEVICE))
model.eval()

print("Model loaded successfully with", len(CLASS_NAMES), "output classes.")

# ---------------------------
# Image transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# FASTAPI
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Helper: preprocess image
# ---------------------------
def read_image(file_bytes):
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    return transform(img).unsqueeze(0).to(DEVICE)

# ---------------------------
# Prediction route
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        input_tensor = read_image(img_bytes)

        with torch.no_grad():
            output = model(input_tensor, BEST_ARCH)
            probs = torch.softmax(output, dim=1)
            confidence, idx = torch.max(probs, 1)

        prediction = CLASS_NAMES[idx.item()]

        return {
            "prediction": prediction,
            "confidence": float(confidence.item())
        }

    except Exception as e:
        return {"error": str(e)}
