"""
gesture_recorder.py — Enregistre des séquences de 30 frames pour les gestes dynamiques
Compatible mediapipe 0.10.33+ (nouvelle API tasks)

Usage :
    python3 src/gesture_recorder.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# Config 
SEQUENCE_LENGTH   = 30        # nb de frames par séquence
SEQUENCES_PER_WORD = 100      # nb de séquences par mot
COUNTDOWN         = 3         # secondes avant enregistrement
DATA_PATH         = "data/sequences"
MODEL_PATH        = "hand_landmarker.task"

# Mots dynamiques à enregistrer
WORDS = ["bonjour", "merci", "sil_vous_plait", "comment", "aide"]

#  Init MediaPipe 
options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
detector = HandLandmarker.create_from_options(options)


def ensure_dirs(word):
    """Crée les dossiers pour chaque mot."""
    path = os.path.join(DATA_PATH, word)
    os.makedirs(path, exist_ok=True)
    return path


def extract_landmarks(hand_landmarks) -> np.ndarray:
    """Transforme les 21 landmarks en 63 valeurs normalisées."""
    wrist = hand_landmarks[0]
    coords = []
    for lm in hand_landmarks:
        coords.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return np.array(coords)


def draw_landmarks(frame, hand_landmarks):
    """Dessine les 21 points sur le frame."""
    h, w, _ = frame.shape
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 100), -1)


def record_sequence(word, seq_num, cap) -> np.ndarray | None:
    """
    Enregistre une séquence de SEQUENCE_LENGTH frames pour un mot.
    Retourne un tableau numpy de shape (30, 63) ou None si raté.
    """
    sequence = []

    for frame_num in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret:
            return None

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        if result.hand_landmarks:
            hand   = result.hand_landmarks[0]
            coords = extract_landmarks(hand)
            draw_landmarks(frame, hand)
        else:
            # Si pas de main détectée → zéros
            coords = np.zeros(63)

        sequence.append(coords)

        # Affichage
        cv2.putText(frame, f"{word} — seq {seq_num+1}/{SEQUENCES_PER_WORD}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame {frame_num+1}/{SEQUENCE_LENGTH}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Barre de progression de la séquence
        progress = int((frame_num / SEQUENCE_LENGTH) * 400)
        cv2.rectangle(frame, (10, 440), (10 + progress, 460), (0, 255, 100), -1)
        cv2.rectangle(frame, (10, 440), (410, 460), (255, 255, 255), 2)

        cv2.imshow("Gesture Recorder — Signum HandSpeak", frame)
        cv2.waitKey(1)

    return np.array(sequence)  # shape: (30, 63)


def record_word(word: str, cap):
    """Enregistre toutes les séquences pour un mot."""
    path = ensure_dirs(word)

    # Compte à rebours initial
    print(f"\n📸 Mot : '{word}' — {SEQUENCES_PER_WORD} séquences de {SEQUENCE_LENGTH} frames")
    print(f"   Fais le signe en mouvement naturellement, encore et encore.")
    print(f"   Tu as {COUNTDOWN}s pour te préparer...")

    start = time.time()
    while time.time() - start < COUNTDOWN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        remaining = int(COUNTDOWN - (time.time() - start)) + 1
        cv2.putText(frame, f"Prepare : {word}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, str(remaining), (280, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 100, 255), 6)
        cv2.imshow("Gesture Recorder — Signum HandSpeak", frame)
        cv2.waitKey(1)

    # Enregistrement des séquences
    for seq_num in range(SEQUENCES_PER_WORD):
        sequence = record_sequence(word, seq_num, cap)

        if sequence is not None:
            # Sauvegarde en fichier .npy
            save_path = os.path.join(path, f"{seq_num}.npy")
            np.save(save_path, sequence)

        # Petite pause entre séquences
        time.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("   Arret demande.")
            return False

    print(f"   '{word}' enregistre ({SEQUENCES_PER_WORD} séquences)")
    return True


def build_gesture_dataset(words: list):
    """Lance l'enregistrement pour chaque mot."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam inaccessible.")
        return

    print("=== Gesture Recorder — Signum HandSpeak ===")
    print(f"Mots a enregistrer : {words}")
    input("Appuie sur Entree pour commencer...")

    for word in words:
        ok = record_word(word, cap)
        if not ok:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSequences sauvegardees dans : {DATA_PATH}/")


if __name__ == "__main__":
    print("=== Gesture Recorder — Signum HandSpeak ===")
    print("1. Mots par défaut :", WORDS)
    print("2. Mots personnalisés")
    choix = input("Ton choix (1 ou 2) : ").strip()

    if choix == "1":
        words = WORDS
    else:
        saisie = input("Entre tes mots séparés par des virgules : ")
        words = [w.strip() for w in saisie.split(',')]

    build_gesture_dataset(words)
