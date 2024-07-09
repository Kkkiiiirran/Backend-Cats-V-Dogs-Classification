from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.models import load_model
import numpy as np
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Successful"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the saved model once during initialization
model = load_model('./model.h5')

def predict_image(file: bytes):
    # Load and preprocess the image
    image = Image.open(io.BytesIO(file))
    image = image.resize((256, 256))
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
