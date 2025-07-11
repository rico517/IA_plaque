import os
import sqlite3
from flask import Flask, request, redirect, url_for, render_template
from datetime import datetime
import cv2
import pytesseract

# Config
UPLOAD_FOLDER = 'uploads'
DB_PATH = 'ocr_data.db'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Init dossier
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Init DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS ocr_resultats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom_image TEXT,
            texte_extrait TEXT,
            date_extraction TEXT
        )
    ''')
    conn.commit()
    conn.close()

# OCR + Enregistrement
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extraire_texte(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    texte = pytesseract.image_to_string(thresh, lang='fra')  # ou 'eng'
    return texte.strip()

def enregistrer_texte(nom_image, texte):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO ocr_resultats (nom_image, texte_extrait, date_extraction)
        VALUES (?, ?, ?)
    ''', (nom_image, texte, datetime.now().isoformat()))
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    texte = None
    if request.method == 'POST':
        fichier = request.files.get('image')
        if fichier and allowed_file(fichier.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], fichier.filename)
            fichier.save(filepath)
            texte = extraire_texte(filepath)
            enregistrer_texte(fichier.filename, texte)
    return render_template('index.html', texte=texte)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
