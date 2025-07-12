"""
Ce script génère des images d'exemple pour un jeu de données d'apprentissage automatique.
Il crée deux ensembles d'images :
1. Images avec texte : Quatre images de 200x100 pixels contenant le texte "Texte 1" à "Texte 4" en noir sur fond blanc. Ces images sont enregistrées dans le dossier 'text'.
2. Images sans texte : Quatre images de 200x100 pixels remplies de valeurs RGB aléatoires. Ces images sont enregistrées dans le dossier 'no_text'.
Dépendances :
- pathlib
- PIL (Pillow)
- numpy
Les dossiers 'text' et 'no_text' doivent exister à la racine du projet avant d'exécuter ce script.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

project_root = Path(".")
text_dir = project_root / "text"
no_text_dir = project_root / "pas_text"

for i in range(4):
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    text = f"Texte {i+1}"
    draw.text((10, 40), text, fill=(0, 0, 0))
    img.save(text_dir / f"text_{i+1}.png")

for i in range(4):
    arr = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(no_text_dir / f"no_text_{i+1}.png")