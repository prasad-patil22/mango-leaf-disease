import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests

app = Flask(__name__)

# =========================
# Download Model from Hugging Face
# =========================

MODEL_PATH = "mango_leaf_model.h5"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Hugging Face...")

    MODEL_URL = (
        "https://huggingface.co/patilprasad/"
        "mango-leaf-disease-model/resolve/main/"
        "mango_leaf_model.h5"
    )

    response = requests.get(MODEL_URL, stream=True)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Model downloaded successfully.")

# =========================
# Upload Folder
# =========================

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# Load Model
# =========================

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")
print("Input Shape:", model.input_shape)

# =========================
# Class Labels
# =========================

classes = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

# =========================
# Routes
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return render_template(
            "index.html",
            prediction="No file selected"
        )

    file = request.files["image"]

    if file.filename == "":
        return render_template(
            "index.html",
            prediction="No file selected"
        )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Preprocess Image
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)

    # Do NOT divide by 255
    # Your model already contains Rescaling(1./255)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    result = classes[predicted_index]

    confidence = float(
        np.max(prediction) * 100
    )

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        image=file.filename
    )

# =========================
# Main
# =========================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )