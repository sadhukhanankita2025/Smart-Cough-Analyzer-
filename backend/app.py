import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import base64
import io
import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# =========================
# CONFIGURATION & MODEL LOADING
# =========================
MODEL_PATH = "cough_model.tflite"
CLASS_NAMES = ["COVID", "Healthy"]  # CRITICAL: If results are reversed, swap to ["Healthy", "COVID"]

interpreter = None
input_details = None
output_details = None
IMG_HEIGHT, IMG_WIDTH = None, None

def init_model():
    global interpreter, input_details, output_details, IMG_HEIGHT, IMG_WIDTH
    if os.path.exists(MODEL_PATH):
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        shape = input_details[0]["shape"]
        IMG_HEIGHT, IMG_WIDTH = shape[1], shape[2]
        print(f"Model Loaded. Expected Input: {shape}")
    else:
        print("Model file not found!")

init_model()

# =========================
# ROBUST FEATURE EXTRACTION
# =========================
def extract_features(file):
    file.seek(0)
    audio_bytes = io.BytesIO(file.read())
    
    # 1. Load with Librosa
    y, sr = librosa.load(audio_bytes, sr=22050, mono=True)
    
    if y is None or len(y) == 0:
        raise ValueError("Empty audio file")

    # 2. STABILITY FIX: Trim leading and trailing silence
    y, _ = librosa.effects.trim(y, top_db=20)

    # 3. STABILITY FIX: Normalize volume (Automatic Gain Control)
    # This ensures the model sees the 'pattern' regardless of recording volume
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # 4. Generate Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_HEIGHT)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # 5. Fixed Width Handling (Padding or Clipping)
    if mel_db.shape[1] < IMG_WIDTH:
        pad_width = IMG_WIDTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :IMG_WIDTH]

    return mel_db, y, sr

# =========================
# VISUALIZATION UTILS
# =========================
def get_encoded_img(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    plt.clf()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def create_visuals(mel_db, y, sr):
    # Spectrogram
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")
    ax1.axis("off")
    spec_base64 = get_encoded_img(fig1)

    # Waveform
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax2)
    ax2.set_title("Normalized Waveform")
    wave_base64 = get_encoded_img(fig2)

    return spec_base64, wave_base64

# =========================
# API ROUTES
# =========================
@app.route("/")
def health():
    return jsonify({"status": "ready", "model": "loaded" if interpreter else "failed"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files["file"]
        
        # Process Audio
        mel_db, y, sr = extract_features(file)
        spec_img, wave_img = create_visuals(mel_db, y, sr)

        # Prepare for Model
        # Scale to [0, 1] or [-1, 1] based on how your model was trained
        input_data = mel_db.astype(np.float32)
        # Simple min-max scaling to stabilize input
        input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data) + 1e-8)

        # Handle Channels
        if input_details[0]["shape"][-1] == 3:
            input_data = np.stack([input_data] * 3, axis=-1)
        else:
            input_data = np.expand_dims(input_data, axis=-1)

        input_data = np.expand_dims(input_data, axis=0)

        # Run Inference
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        # Determine Result
        if len(output) == 1:
            prob = float(output[0])
            class_idx = 1 if prob > 0.5 else 0
            conf = prob if prob > 0.5 else (1 - prob)
        else:
            class_idx = int(np.argmax(output))
            conf = float(np.max(output))

        label = CLASS_NAMES[class_idx]
        
        return jsonify({
            "label": label,
            "confidence": round(conf * 100, 2),
            "spectrogram": spec_img,
            "waveform": wave_img,
            "recommendation": "Consult a doctor if symptoms persist." if label == "COVID" else "Stay healthy and hydrated.",
            "suggestions": ["Rest", "Mask up", "Hydrate"] if label == "COVID" else ["Exercise", "Healthy diet"]
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Disable reloader to prevent TFLite memory locks
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)