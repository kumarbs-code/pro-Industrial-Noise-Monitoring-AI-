import hashlib
import os

# Reduce OpenBLAS/BLAS threading to avoid excessive memory usage on constrained environments.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from spectrogram import audio_to_spectrogram

app = Flask(__name__)

# model = tf.keras.models.load_model("industrial_noise_model.h5")

classes = ["Alarm", "Metal Hit", "Normal Machine", "Silence"]

HERE = os.path.dirname(__file__)
SAMPLES_DIR = os.path.join(HERE, "static", "samples")


def list_samples() -> list[str]:
    """Return the list of available example audio files."""
    try:
        return sorted([f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".wav")])
    except FileNotFoundError:
        return []


def fake_prediction_scores(audio_path: str) -> list[float]:
    """Produce a deterministic pseudo-confidence vector from a file hash."""
    try:
        with open(audio_path, "rb") as f:
            data = f.read(4096)
    except FileNotFoundError:
        data = b""

    digest = hashlib.sha256(data).digest()
    values = np.frombuffer(digest, dtype=np.uint8).astype(float)
    # Ensure we have enough values for all classes
    if len(values) < len(classes):
        values = np.tile(values, int(np.ceil(len(classes) / len(values))))

    values = values[: len(classes)]
    probs = values / values.sum() if values.sum() > 0 else np.ones(len(classes)) / len(classes)
    return probs.tolist()


def analyze_audio_file(audio_path: str) -> dict:
    """Convert audio to spectrogram + make a prediction (dummy if model missing)."""
    image_path = audio_to_spectrogram(audio_path)

    img = Image.open(image_path).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # prediction = model.predict(img)
    scores = fake_prediction_scores(audio_path)
    prediction = classes[int(np.argmax(scores))]

    return {"prediction": prediction, "scores": scores, "classes": classes}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/record", methods=["POST"])
def record():
    """Accepts a recorded audio blob from the browser and returns prediction JSON."""
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No audio file provided"}), 400

    audio_path = "uploads/record.wav"
    audio.save(audio_path)

    result = analyze_audio_file(audio_path)
    return jsonify(result)


@app.route("/upload", methods=["POST"])
def upload():
    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No file uploaded"}), 400

    audio_path = "uploads/upload.wav"
    audio.save(audio_path)

    result = analyze_audio_file(audio_path)
    return jsonify(result)


@app.route("/samples")
def samples():
    return jsonify({"samples": list_samples()})


@app.route("/sample/<name>")
def sample(name: str):
    if ".." in name or name.startswith("/"):
        return jsonify({"error": "Invalid sample name"}), 400

    sample_path = os.path.join(SAMPLES_DIR, name)
    if not os.path.isfile(sample_path):
        return jsonify({"error": "Sample not found"}), 404

    result = analyze_audio_file(sample_path)
    return jsonify(result)


if __name__ == "__main__":
    # Choose port from environment for hosting platforms (Render, Railway, etc.)
    port = int(os.environ.get("PORT", 5000))
    # Disable debugger/reloader to reduce memory usage in constrained environments
    app.run(host="0.0.0.0", port=port, debug=False)
