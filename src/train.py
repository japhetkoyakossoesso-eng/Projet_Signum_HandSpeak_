"""
train.py — Entraîne un modèle Random Forest sur le dataset CSV
Compatible mediapipe 0.10.33+ (nouvelle API tasks)

Usage :
    python3 src/train.py
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_PATH = "data/landmarks/dataset.csv"
MODEL_PATH   = "models/alphabet_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
HAND_MODEL   = "hand_landmarker.task"


def load_data():
    """Charge et prépare le dataset."""
    print("📂 Chargement du dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"   {len(df)} échantillons | {df['label'].nunique()} classes")
    print(f"   Classes : {sorted(df['label'].unique())}")
    X = df.drop('label', axis=1).values
    y = df['label'].values
    return X, y


def train(X, y):
    """Encode les labels, split, entraîne et évalue."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("\n🧠 Entraînement du Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"\n✅ Précision globale : {acc*100:.1f}%")
    print("\n📊 Rapport par classe :")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Matrice de confusion")
    plt.ylabel("Réel")
    plt.xlabel("Prédit")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    print("   📈 Matrice sauvegardée dans models/confusion_matrix.png")

    return model, le


def save_model(model, le):
    """Sauvegarde le modèle et l'encodeur."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"\n💾 Modèle sauvegardé : {MODEL_PATH}")
    print(f"💾 Encodeur sauvegardé : {ENCODER_PATH}")


def extract_landmarks(hand_landmarks) -> list:
    """Transforme les 21 landmarks en 63 valeurs normalisées."""
    wrist = hand_landmarks[0]
    coords = []
    for lm in hand_landmarks:
        coords.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return coords


def predict_realtime():
    """Test du modèle en temps réel après entraînement."""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    print("\n🎥 Test temps réel — Q pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        if result.hand_landmarks:
            hand   = result.hand_landmarks[0]
            coords = extract_landmarks(hand)

            pred_idx = model.predict([coords])[0]
            proba    = model.predict_proba([coords])[0].max()
            label    = le.inverse_transform([pred_idx])[0]

            # Dessine les points
            h, w, _ = frame.shape
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 100), -1)

            # Affiche la lettre + confiance
            cv2.putText(frame, f"{label}  ({proba*100:.0f}%)", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 100), 4)

        cv2.imshow("Test modele — Signum HandSpeak", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    X, y = load_data()
    model, le = train(X, y)
    save_model(model, le)

    test = input("\n🎥 Tester le modèle en temps réel ? (o/n) : ").strip().lower()
    if test == 'o':
        predict_realtime()