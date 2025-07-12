"""
Application web Flask pour détecter et extraire le texte des images téléchargées.

Routes :
    - "/" (GET, POST) : Page principale pour télécharger des images. En POST, sauvegarde l'image,
      extrait des caractéristiques, prédit si du texte est présent avec un modèle de machine learning,
      et extrait le texte avec pytesseract si du texte est détecté. Affiche les résultats.

Fonctions :
    - extract_features(image_path) : Charge une image, la redimensionne, applique un seuillage binaire,
      et calcule le ratio de pixels blancs comme caractéristique pour la détection de texte.

Configuration :
    - UPLOAD_FOLDER : Dossier pour stocker les images téléchargées.
    - model : Modèle scikit-learn pré-entraîné pour la détection de texte chargé depuis 'ml/text_detector_model.pkl'.

Dépendances :
    - Flask, werkzeug, cv2 (OpenCV), pytesseract, joblib, numpy
"""

from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import pytesseract
import joblib
import numpy as np
from database import init_db, insert_upload, get_all_uploads

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialiser la base de données
init_db()

# Charger le modèle
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
            features = extract_features(filepath)
            pred = model.predict([features])[0]

            if pred == 0:
                extracted_text = "[Aucun texte détecté]"
            else:
                img = cv2.imread(filepath)
                extracted_text = pytesseract.image_to_string(img)

                insert_upload(filename, filepath, pred, extracted_text)

            return render_template("result.html", text=extracted_text)
    return render_template("index.html")

@app.route('/history')
def history():
    uploads = get_all_uploads()
    return render_template('history.html', uploads=uploads)


if __name__ == "__main__":
    app.run(debug=True)