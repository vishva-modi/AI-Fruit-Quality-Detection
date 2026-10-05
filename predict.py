import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("model/fruit_quality_model.keras")

class_names = [
    "fresh_apple",
    "fresh_banana",
    "fresh_orange",
    "rotten_apple",
    "rotten_banana",
    "rotten_orange"
]

img_path = "test_image.jpg"

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

fruit, quality = predicted_class.split("_")

print("\n--- Fruit Quality Detection Result ---")
print("Fruit:", fruit.capitalize())
print("Quality:", quality.capitalize())
print("Confidence:", f"{confidence:.2f}%")