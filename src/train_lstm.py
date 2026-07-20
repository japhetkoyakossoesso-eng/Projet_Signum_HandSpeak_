"""
train_lstm.py — Entraîne un modèle LSTM sur les séquences de gestes dynamiques
Compatible Python 3.13 — utilise scikit-learn LSTM-like avec séquences numpy

Usage :
    python3 src/train_lstm.py
"""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow / Keras 
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} détecté")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow non disponible — utilisation du modèle alternatif")

# Config
DATA_PATH    = "data/sequences"
MODEL_PATH   = "models/words_model"
ENCODER_PATH = "models/words_encoder.pkl"
SEQUENCE_LENGTH = 30


def load_sequences():
    """Charge toutes les séquences .npy depuis data/sequences/."""
    print(" Chargement des séquences...")

    words = sorted(os.listdir(DATA_PATH))
    words = [w for w in words if os.path.isdir(os.path.join(DATA_PATH, w))]

    X, y = [], []

    for word in words:
        word_path = os.path.join(DATA_PATH, word)
        files = sorted([f for f in os.listdir(word_path) if f.endswith('.npy')])

        for f in files:
            seq = np.load(os.path.join(word_path, f))
            if seq.shape == (SEQUENCE_LENGTH, 63):
                X.append(seq)
                y.append(word)

    X = np.array(X)
    y = np.array(y)

    print(f"   {len(X)} séquences | {len(set(y))} classes")
    print(f"   Classes : {sorted(set(y))}")
    print(f"   Shape X : {X.shape}")

    return X, y, words


def train_with_tensorflow(X, y, words):
    """Entraîne le modèle LSTM avec TensorFlow/Keras."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc, num_classes=len(words))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )

    print("\n🧠 Construction du modèle LSTM...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 63)),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(words), activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(patience=7, factor=0.5),
    ]

    print("\n🏋️ Entraînement...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # Évaluation
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Précision : {acc*100:.1f}%")

    # Rapport détaillé
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    print(classification_report(y_test_classes, y_pred_classes, target_names=le.classes_))

    # Matrice de confusion
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Matrice de confusion — LSTM")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix_lstm.png", dpi=150)

    # Courbes d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Précision')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("models/training_curves.png", dpi=150)
    print("   📈 Courbes sauvegardées dans models/")

    return model, le


def save_model_tf(model, le):
    """Sauvegarde le modèle TensorFlow et l'encodeur."""
    os.makedirs("models", exist_ok=True)
    model.save(f"{MODEL_PATH}.keras")
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"\n💾 Modèle LSTM sauvegardé : {MODEL_PATH}.keras")
    print(f"💾 Encodeur sauvegardé : {ENCODER_PATH}")


def predict_realtime_lstm():
    """Test du modèle LSTM en temps réel."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    from collections import deque

    if TF_AVAILABLE:
        model = tf.keras.models.load_model(f"{MODEL_PATH}.keras")
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    print("\n🎥 Test LSTM en temps réel — Q pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            wrist = hand[0]
            coords = []
            for lm in hand:
                coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            sequence.append(np.array(coords))

            h, w, _ = frame.shape
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 100), -1)

        # Prédiction quand séquence complète
        if len(sequence) == SEQUENCE_LENGTH:
            X_pred = np.expand_dims(np.array(sequence), axis=0)
            pred   = model.predict(X_pred, verbose=0)[0]
            idx    = np.argmax(pred)
            label  = le.inverse_transform([idx])[0]
            proba  = pred[idx]

            if proba > 0.7:
                cv2.putText(frame, f"{label} ({proba*100:.0f}%)", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 100), 3)

        # Barre de remplissage séquence
        progress = int((len(sequence) / SEQUENCE_LENGTH) * 300)
        cv2.rectangle(frame, (10, 440), (10 + progress, 460), (0, 200, 255), -1)
        cv2.rectangle(frame, (10, 440), (310, 460), (255, 255, 255), 2)

        cv2.imshow("LSTM Test — Signum HandSpeak", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    X, y, words = load_sequences()

    if len(X) == 0:
        print("❌ Aucune séquence trouvée — lance d'abord gesture_recorder.py")
    elif TF_AVAILABLE:
        model, le = train_with_tensorflow(X, y, words)
        save_model_tf(model, le)

        test = input("\n🎥 Tester en temps réel ? (o/n) : ").strip().lower()
        if test == 'o':
            predict_realtime_lstm()
    else:
        print("❌ TensorFlow requis pour le LSTM — installe-le avec : pip install tensorflow")
