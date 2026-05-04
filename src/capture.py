import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle MediaPipe...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modèle téléchargé ")

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
detector = HandLandmarker.create_from_options(options)

def extract_landmarks(hand_landmarks) -> list:
    wrist = hand_landmarks[0]
    coords = []
    for lm in hand_landmarks:
        coords.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return coords

def run_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam.")
        return

    print("Webcam lancée, appuie sur Q pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                coords = extract_landmarks(hand)
                print(f"Landmarks : {coords[:6]} ...")

                h, w, _ = frame.shape
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 100), -1)

        cv2.imshow("Sign Language — Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_capture()