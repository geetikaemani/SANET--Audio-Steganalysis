from flask import Flask, render_template, send_from_directory

import os

# Detect frontend directory automatically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# ---------------------------- #
#         ROUTES              #
# ---------------------------- #

@app.route("/")
def home():
    return render_template("index.html")

# (Optional) Serve uploaded files if needed later
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    upload_folder = os.path.join(BASE_DIR, "uploads")
    return send_from_directory(upload_folder, filename)


if __name__ == "__main__":
    print(f"📁 Templates: {TEMPLATE_DIR}")
    print(f"🎨 Static: {STATIC_DIR}")
    print("🚀 Frontend loaded at: http://127.0.0.1:5000")
    app.run(debug=True)
