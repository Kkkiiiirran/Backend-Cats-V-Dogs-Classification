from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Model is ready to classify images"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Download the model from Hugging Face
MODEL_URL = "https://huggingface.co/spaces/Sa-m/Dogs-vs-Cats/resolve/main/best_model.h5"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("Model downloaded successfully.")

# Load the model
model = load_model(MODEL_PATH)

def predict_image(file: bytes):
    # Load and preprocess the image
    image = Image.open(io.BytesIO(file))
    image = image.resize((150, 150))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image to match training data

    # Predict using the model
    prediction = model.predict(img_array)
    return prediction

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        prediction = predict_image(file_bytes)
        class_name = "dog" if prediction[0][0] > 0.5 else "cat"
        class_id = 1 if prediction[0][0] > 0.5 else 0
        return JSONResponse(content={"class_name": class_name, "class": class_id})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
