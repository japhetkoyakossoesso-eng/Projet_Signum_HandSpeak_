"""
dataset_builder.py — Enregistre les gestes et crée le dataset CSV
Compatible mediapipe 0.10.33+ (nouvelle API tasks)

Usage :
    python3 src/dataset_builder.py
"""

import cv2
import mediapipe as mp
import csv
import os
import time

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

DATASET_PATH      = "data/landmarks/dataset.csv"
MODEL_PATH        = "hand_landmarker.task"
SAMPLES_PER_LABEL = 200
COUNTDOWN         = 3


options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
detector = HandLandmarker.create_from_options(options)


def ensure_dirs():
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)


def extract_landmarks(hand_landmarks) -> list:
    """Transforme les 21 landmarks en 63 valeurs normalisées [x,y,z]."""
    wrist = hand_landmarks[0]
    coords = []
    for lm in hand_landmarks:
        coords.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return coords


def draw_landmarks(frame, hand_landmarks):
    """Dessine les 21 points sur le frame."""
    h, w, _ = frame.shape
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 100), -1)


def record_gesture(label: str, cap, writer) -> bool:
    """Enregistre SAMPLES_PER_LABEL échantillons pour un geste donné."""
    print(f"\n Prépare-toi pour : '{label}'")
    print(f"   Tu as {COUNTDOWN}s pour positionner ta main...")

    start = time.time()
    while time.time() - start < COUNTDOWN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        remaining = int(COUNTDOWN - (time.time() - start)) + 1
        cv2.putText(frame, f"Prepare : {label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        cv2.putText(frame, str(remaining), (280, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 100, 255), 6)
        cv2.imshow("Dataset Builder", frame)
        cv2.waitKey(1)

    count = 0
    print(f"   Enregistrement en cours ({SAMPLES_PER_LABEL} samples)...")

    while count < SAMPLES_PER_LABEL:
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
            writer.writerow([label] + coords)
            count += 1
            draw_landmarks(frame, hand)

        progress = int((count / SAMPLES_PER_LABEL) * 400)
        cv2.rectangle(frame, (20, 430), (20 + progress, 460), (0, 255, 0), -1)
        cv2.rectangle(frame, (20, 430), (420, 460), (255, 255, 255), 2)
        cv2.putText(frame, f"{label}  {count}/{SAMPLES_PER_LABEL}", (20, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Dataset Builder", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("   Arret demande.")
            return False

    print(f"   '{label}' enregistre ({count} samples)")
    return True


def build_dataset(labels: list):
    ensure_dirs()

    file_exists = os.path.exists(DATASET_PATH)
    mode = 'a' if file_exists else 'w'

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam inaccessible.")
        return

    with open(DATASET_PATH, mode, newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]
            writer.writerow(header)

        for label in labels:
            ok = record_gesture(label, cap, writer)
            if not ok:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDataset sauvegarde dans : {DATASET_PATH}")


if __name__ == "__main__":
    import string

    print("Dataset Builder — Signum HandSpeak")
    print("1. Alphabet (A-Z)")
    print("2. Mots personnalises")
    choix = input("Ton choix (1 ou 2) : ").strip()

    if choix == "1":
        labels = list(string.ascii_uppercase)
    else:
        saisie = input("Entre tes gestes separes par des virgules (ex: bonjour,merci,oui) : ")
        labels = [l.strip() for l in saisie.split(',')]

    print(f"\nGestes a enregistrer : {labels}")
    input("Appuie sur Entree pour commencer...")

    build_dataset(labels)