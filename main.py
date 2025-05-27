from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import requests

MODEL_URL = "https://huggingface.co/spaces/Tauqueer-Alam/Dog-Vs-Cat-Classifier/resolve/main/model.h5"
MODEL_PATH = "model.h5"

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "API is live"}

def download_model():
    """Download model from Hugging Face if not present locally."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model.h5 from Hugging Face...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded.")

# Download model if needed and load it
download_model()
model = load_model(MODEL_PATH)

def predict_image(file: bytes):
    """Preprocess and predict"""
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = image.resize((256, 256))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)
    return prediction

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        prediction = predict_image(file_bytes)
        class_name = "dog" if prediction[0][0] > 0.5 else "cat"
        class_id = 1 if prediction[0][0] > 0.5 else 0
        return {"class_name": class_name, "class": class_id}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
