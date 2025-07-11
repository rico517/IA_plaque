# OCR WebApp

Une application web simple en Python/Flask pour :

- Uploader une image via une interface web
- Extraire le texte de l’image via OCR (Tesseract)
- Enregistrer les résultats dans une base de données SQLite

## 🚀 Démarrage

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Installer Tesseract OCR

#### Ubuntu/Debian :
```bash
sudo apt install tesseract-ocr
```

#### macOS :
```bash
brew install tesseract
```

#### Windows :
Télécharger depuis : https://github.com/tesseract-ocr/tesseract

### 3. Lancer l'application

```bash
python app.py
```

Ouvre ensuite [http://localhost:5000](http://localhost:5000) dans ton navigateur.
