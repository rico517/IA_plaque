
import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    white_pixel_ratio = np.sum(thresh == 255) / (100 * 100)
    return [white_pixel_ratio]

def build_dataset(data_dir):
    X, y = [], []
    for label in ['text', 'no_text']:
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(label_dir, fname)
                features = extract_features(fpath)
                X.append(features)
                y.append(1 if label == 'text' else 0)
    return np.array(X), np.array(y)

def train_and_save_model(data_dir, model_path):
    X, y = build_dataset(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, model_path)

if __name__ == "__main__":
    train_and_save_model("ml/training_data", "ml/text_detector_model.pkl")
