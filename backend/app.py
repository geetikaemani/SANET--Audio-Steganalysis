import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from inference import run_prediction  # Only import run_prediction

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "sanet-frontend")
TEMPLATES_DIR = os.path.join(FRONTEND_DIR, "templates")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Create Flask app
app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)


@app.route("/")
def home():
    """Serve frontend UI"""
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """Receive file → run inference → return JSON"""
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    result = run_prediction(save_path)

    return jsonify(result)


@app.route("/results/<path:filename>")
def serve_results(filename):
    """Serve generated spectrogram images"""
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    print("🚀 Backend started at: http://127.0.0.1:5000")
    print(f"📁 Templates loaded from: {TEMPLATES_DIR}")
    print(f"🎨 Static loaded from:    {STATIC_DIR}")
    app.run(debug=True)
