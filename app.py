
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import pytesseract
import joblib
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = joblib.load('ml/text_detector_model.pkl')

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    white_pixel_ratio = np.sum(thresh == 255) / (100 * 100)
    return [white_pixel_ratio]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Prediction: y a-t-il du texte ?
            features = extract_features(filepath)
            pred = model.predict([features])[0]

            if pred == 0:
                extracted_text = "[Aucun texte détecté]"
            else:
                img = cv2.imread(filepath)
                extracted_text = pytesseract.image_to_string(img)

            return render_template("result.html", text=extracted_text)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
