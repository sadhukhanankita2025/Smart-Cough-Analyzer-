import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "cough_model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SHAPE = input_details[0]['shape']
IMG_HEIGHT = INPUT_SHAPE[1]
IMG_WIDTH = INPUT_SHAPE[2]

CLASS_NAMES = ["Healthy", "COVID"]

# =========================
# UI FIRST
# =========================
st.title("🩺 Smart Cough Analyzer")

# 👉 DEFINE FILE HERE
file = st.file_uploader("Upload WAV file", type=["wav"])

# =========================
# FUNCTION
# =========================
def extract_features(file):
    y, sr = librosa.load(file, sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_HEIGHT)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < IMG_WIDTH:
        pad = IMG_WIDTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad)))
    else:
        mel_db = mel_db[:, :IMG_WIDTH]

    return mel_db

# =========================
# USE FILE AFTER DEFINITION
# =========================
if file is not None:

    st.audio(file)

    try:
        features = extract_features(file)

        if np.max(features) != 0:
            features = features / np.max(features)

        if INPUT_SHAPE[3] == 3:
            features = np.stack([features]*3, axis=-1)
        else:
            features = np.expand_dims(features, axis=-1)

        features = np.expand_dims(features, axis=0).astype(np.float32)

        # Prediction
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        # Output logic
        if len(output) == 1:
            confidence = float(output[0])
            class_id = 1 if confidence > 0.5 else 0
        else:
            class_id = np.argmax(output)
            confidence = float(np.max(output))

        label = CLASS_NAMES[class_id]

        # Result
        st.subheader("🧾 Result")
        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        # Recommendation
        st.subheader("💡 Recommendation")
        if label == "COVID":
            st.error("Consult doctor, isolate, take test")
        else:
            st.success("You seem healthy")

        # Graph
        st.subheader("📊 Spectrogram")
        fig, ax = plt.subplots()
        ax.imshow(features[0][:,:,0], aspect='auto', origin='lower')
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}") 