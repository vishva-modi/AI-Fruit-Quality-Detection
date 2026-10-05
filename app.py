from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json
from datetime import datetime

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/fruit_quality_model.keras")

# Class names used during training
class_names = [
    "fresh_apple",
    "fresh_banana",
    "fresh_orange",
    "rotten_apple",
    "rotten_banana",
    "rotten_orange"
]

# Upload folder
upload_folder = "static/uploads"
history_file = "history.json"

os.makedirs(upload_folder, exist_ok=True)


# -----------------------------
# History Functions
# -----------------------------

def load_history():
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as file:
                return json.load(file)
        except:
            return []

    return []


def save_history(data):
    with open(history_file, "w") as file:
        json.dump(data, file, indent=4)


# -----------------------------
# Home Page
# -----------------------------

@app.route("/")
def home():

    history = load_history()

    return render_template(
        "index.html",
        history=history[-5:]
    )


# -----------------------------
# Prediction
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("image")

    if not file or file.filename == "":
        return "Please select an image."

    # Save uploaded image
    file_path = os.path.join(
        upload_folder,
        "uploaded_fruit.jpg"
    )

    file.save(file_path)

    # Load image
    img = image.load_img(
        file_path,
        target_size=(128, 128)
    )

    # Convert image to array
    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Normalize pixel values
    img_array = img_array / 255.0

    # Model prediction
    prediction = model.predict(
        img_array,
        verbose=0
    )

    # Get predicted class
    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    # Confidence
    confidence = float(
        np.max(prediction) * 100
    )

    # Split quality and fruit
    quality, fruit = predicted_class.split("_")

    # Terminal debugging information
    print("\n-----------------------------")
    print("RAW PREDICTION:", prediction)
    print("PREDICTED CLASS:", predicted_class)
    print("FRUIT:", fruit)
    print("QUALITY:", quality)
    print("CONFIDENCE:", f"{confidence:.2f}%")
    print("-----------------------------\n")

    # Save prediction history
    history = load_history()

    history.append({
        "fruit": fruit.capitalize(),
        "quality": quality.capitalize(),
        "confidence": round(confidence, 2),
        "date": datetime.now().strftime(
            "%d-%m-%Y %I:%M %p"
        )
    })

    save_history(history)

    # Show result page
    return render_template(
        "result.html",
        fruit=fruit.capitalize(),
        quality=quality.capitalize(),
        confidence=f"{confidence:.2f}",
        image_path="/" + file_path.replace("\\", "/")
    )


# -----------------------------
# History Page
# -----------------------------

@app.route("/history")
def history():

    return render_template(
        "history.html",
        history=load_history()
    )
@app.route("/clear-history", methods=["POST"])
def clear_history():

    save_history([])

    return render_template(
        "history.html",
        history=[]
    )


# -----------------------------
# About Page
# -----------------------------

@app.route("/info")
def info():

    return render_template(
        "info.html"
    )


# -----------------------------
# Run Flask App
# -----------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )