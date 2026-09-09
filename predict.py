import os
import numpy as np
import json

# --- CRITICAL FIX FOR LAZY LOADER LOOP ---
# Explicitly import keras components before tensorflow to bypass the recursion bug
import keras
import tensorflow as tf

# ---------------- AUTO-MERGE MODEL CHUNKS ----------------
if not os.path.exists("model.h5"):
    print("🔧 Reconstructing model from uploaded parts...")
    with open("model.h5", "wb") as main_file:
        for part in ["model_part_1.h5", "model_part_2.h5"]:
            if os.path.exists(part):
                with open(part, "rb") as part_file:
                    main_file.write(part_file.read())

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("model.h5")
print("✅ Model loaded successfully!")

# ---------------- LOAD CLASS NAMES (FAST) ----------------
def get_class_names():
    if os.path.exists("class_names.json"):
        with open("class_names.json", "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError("❌ class_names.json not found. Train model first.")

class_names = get_class_names()
print("📂 Classes:", class_names)


# ---------------- PREDICT FUNCTION ----------------
def predict_image(pil_image):
    # Resize & preprocess natively using PIL and NumPy
    img = pil_image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    sorted_indices = np.argsort(prediction)[::-1]

    top1_idx = sorted_indices
    top2_idx = sorted_indices if len(sorted_indices) > 1 else 0

    top1 = prediction[top1_idx]
    top2 = prediction[top2_idx]

    confidence = float(top1)
    predicted_class = class_names[top1_idx]

    # ---------------- TOP 3 ----------------
    top3 = [(class_names[i], float(prediction[i])) for i in sorted_indices[:3]]

    # ---------------- STATUS ----------------
    if predicted_class.lower() == "unknown":
        status = "unknown"
    elif top1 < 0.5 or (top1 - top2) < 0.1:
        status = "uncertain"
    else:
        status = "confident"

    return status, predicted_class, confidence, top3, prediction
