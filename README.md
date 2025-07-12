
# IA_Plaque - OCR et Détection de Texte avec Flask + ML

Ce projet est une application web Flask permettant :
- D'uploader une image via une interface web.
- De détecter automatiquement si l'image contient du texte (grâce à un modèle ML simple).
- D'appliquer de l'OCR (reconnaissance de texte) si le modèle détecte du texte.
- D'afficher le texte extrait dans une page web.

## 🧠 Modèle d'apprentissage
- **Type** : Forêt Aléatoire (`RandomForestClassifier`)
- **Objectif** : Classifier une image comme "texte" ou "pas de texte"
- **Caractéristique utilisée** : Ratio de pixels blancs après seuillage sur une image redimensionnée en 100x100
- **Format d'entrée attendu** : 
  ```
  ml/training_data/
    ├── text/       # images contenant du texte
    └── pas_text/    # images sans texte
  ```

## 🔍 OCR
- Utilise [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) via `pytesseract` pour extraire le texte uniquement si l'image est prédite comme contenant du texte.

## 💻 Interface Web
- Développée avec Flask.
- Interface simple pour charger une image et afficher le résultat.

## 🚀 Lancement

### 1. Entraîner le modèle
```bash
python ml/train_model.py
```

### 2. Lancer l'application Flask
```bash
python app.py
```

Puis ouvrir [http://127.0.0.1:5000](http://127.0.0.1:5000) dans le navigateur.

---

## 📦 Dépendances

Voir `requirements.txt`
