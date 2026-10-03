from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load trained CNN model
model = tf.keras.models.load_model("brain_tumor_cnn.keras")

# Class names
class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

IMG_SIZE = 128

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No image selected"

    # Save image
    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Open image
    image = Image.open(filepath)

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to NumPy
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array)

    # Find class
    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    # Confidence
    confidence = np.max(predictions[0]) * 100

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence, 2),
        image_path=filepath
    )

if __name__ == "__main__":
    app.run()