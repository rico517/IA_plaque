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
import pytesseract
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
from database import init_db, insert_upload, get_all_uploads

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialiser la base de données
init_db()

# Charger le modèle
model = joblib.load('ml/text_detector_model.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    white_ratio = np.sum(thresh == 255) / (100 * 100)
    return [white_ratio]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(save_path)

            features = preprocess_image(save_path)
            prediction = model.predict([features])[0]
            contains_text = int(prediction == 1)

            ocr_result = ""
            if contains_text:
                img_cv = cv2.imread(save_path)
                ocr_result = pytesseract.image_to_string(img_cv)

            insert_upload(filename, save_path, contains_text, ocr_result)

            return render_template('result.html', image=filename, contains_text=contains_text, text=ocr_result)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

@app.route('/history')
def history():
    uploads = get_all_uploads()
    return render_template('history.html', uploads=uploads)

if __name__ == '__main__':
    app.run(debug=True)
