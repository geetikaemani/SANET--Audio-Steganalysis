import os
from flask import Blueprint, request, jsonify, render_template

routes_bp = Blueprint("routes", __name__)

# Folders
UPLOAD_FOLDER = "uploads"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@routes_bp.route("/")
def home():
    return render_template("index.html")

@routes_bp.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("audio")
    if not file:
        return jsonify({"error": "No audio file found"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # --- FAKE RESULT (Will be replaced when SANet is ready) ---
    import random
    label = random.choice(["CLEAN AUDIO", "STEGO DETECTED"])
    confidence = round(random.uniform(75, 99), 2)

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "spectrogram1": "/static/img/dummy_spec_1.png",
        "spectrogram2": "/static/img/dummy_spec_2.png",
        "report": "/reports/dummy_report.pdf"
    })
