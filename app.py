import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# --- Config ---
MODEL_PATH = "model/my_model.keras"
TOKENIZER_PATH = "model/tokenizer.pkl"
MAX_LEN = 35
THRESHOLD = 0.5

# --- Flask App ---
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load model and tokenizer
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# --- Preprocessing Function ---
def preprocess(sam1, sam2):
    docx = [sam1 + " " + sam2]
    sequences = tokenizer.texts_to_sequences(docx)
    sequences = pad_sequences(sequences, padding = "post", maxlen = 35)
    return sequences

# --- Web UI Route ---
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# --- API Route ---
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    q1 = data.get("question1", "").strip()
    q2 = data.get("question2", "").strip()

    if not q1 or not q2:
        return jsonify({"error": "Both question1 and question2 are required."}), 400

    x = preprocess(q1, q2)
    prob = float(model.predict(x, verbose=0)[0][0])
    label = "DUPLICATE" if prob >= THRESHOLD else "DIFFERENT"

    return jsonify({"label": label, "score": prob})

# --- Local Dev ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)

