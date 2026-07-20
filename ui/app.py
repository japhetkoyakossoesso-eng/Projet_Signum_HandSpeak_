"""
app.py — Interface principale Tkinter pour Signum HandSpeak

Usage :
    python3 ui/app.py
"""

import tkinter as tk
import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import time
import os
from PIL import Image, ImageTk
from collections import deque

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# Config 
HAND_MODEL      = "hand_landmarker.task"
ALPHA_MODEL     = "models/alphabet_model.pkl"
ALPHA_ENCODER   = "models/label_encoder.pkl"
LSTM_MODEL      = "models/words_model.keras"
LSTM_ENCODER    = "models/words_encoder.pkl"
SEQUENCE_LENGTH = 30

# Palette
BG_DARK  = "#0d0d1a"
BG_MID   = "#12122a"
BG_PANEL = "#1a1a3a"
ACCENT   = "#00ff88"
ACCENT2  = "#4488ff"
TEXT_PRI = "#e8e8ff"
TEXT_SEC = "#7a7aaa"
DANGER   = "#ff5f57"
WARNING  = "#ffcc00"


class SignumApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signum HandSpeak — Traducteur LSF")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1000x680")
        self.root.resizable(False, False)

        self.running   = False
        self.cap       = None
        self.sequence  = deque(maxlen=SEQUENCE_LENGTH)
        self.history   = []
        self.mode      = tk.StringVar(value="alphabet")

        self._load_models()
        self._init_mediapipe()
        self._build_ui()

    def _load_models(self):
        try:
            with open(ALPHA_MODEL, 'rb') as f:
                self.alpha_model = pickle.load(f)
            with open(ALPHA_ENCODER, 'rb') as f:
                self.alpha_encoder = pickle.load(f)
            self.alpha_ok = True
        except Exception as e:
            print(f"Modèle alphabet non trouvé : {e}")
            self.alpha_ok = False

        try:
            import tensorflow as tf
            self.lstm_model = tf.keras.models.load_model(LSTM_MODEL)
            with open(LSTM_ENCODER, 'rb') as f:
                self.lstm_encoder = pickle.load(f)
            self.lstm_ok = True
        except Exception as e:
            print(f"Modèle LSTM non trouvé : {e}")
            self.lstm_ok = False

    def _init_mediapipe(self):
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.detector = HandLandmarker.create_from_options(options)

    def _build_ui(self):
        # Titlebar
        titlebar = tk.Frame(self.root, bg=BG_MID, height=48)
        titlebar.pack(fill=tk.X)
        titlebar.pack_propagate(False)
        tk.Label(titlebar, text="🤟  SIGNUM HANDSPEAK",
                 bg=BG_MID, fg=ACCENT,
                 font=("Courier", 14, "bold")).pack(side=tk.LEFT, padx=20, pady=12)
        tk.Label(titlebar, text="Traducteur de Langue des Signes Française",
                 bg=BG_MID, fg=TEXT_SEC,
                 font=("Courier", 10)).pack(side=tk.LEFT, pady=12)

        # Layout principal
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Colonne gauche — Webcam
        left = tk.Frame(main, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(left, width=620, height=465,
                                bg="#050510", highlightthickness=1,
                                highlightbackground=BG_PANEL)
        self.canvas.pack()
        self.canvas.create_text(310, 232, text="Appuie sur Démarrer",
                                fill=TEXT_SEC, font=("Courier", 14),
                                tags="placeholder")

        # Colonne droite
        right = tk.Frame(main, bg=BG_DARK, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))
        right.pack_propagate(False)

        # Bloc détection
        detect_frame = tk.Frame(right, bg=BG_MID, pady=16)
        detect_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(detect_frame, text="SIGNE DÉTECTÉ",
                 bg=BG_MID, fg=TEXT_SEC,
                 font=("Courier", 9)).pack()
        self.label_detected = tk.Label(detect_frame, text="—",
                                       bg=BG_MID, fg=ACCENT,
                                       font=("Courier", 56, "bold"))
        self.label_detected.pack()

        conf_container = tk.Frame(detect_frame, bg=BG_MID)
        conf_container.pack(fill=tk.X, padx=20, pady=(4, 0))
        conf_top = tk.Frame(conf_container, bg=BG_MID)
        conf_top.pack(fill=tk.X)
        tk.Label(conf_top, text="CONFIANCE", bg=BG_MID, fg=TEXT_SEC,
                 font=("Courier", 8)).pack(side=tk.LEFT)
        self.label_conf = tk.Label(conf_top, text="0%", bg=BG_MID, fg=ACCENT,
                                   font=("Courier", 8, "bold"))
        self.label_conf.pack(side=tk.RIGHT)
        self.conf_bar_bg = tk.Canvas(conf_container, height=6, bg=BG_PANEL,
                                     highlightthickness=0)
        self.conf_bar_bg.pack(fill=tk.X, pady=(3, 0))
        self.conf_bar = self.conf_bar_bg.create_rectangle(0, 0, 0, 6,
                                                           fill=ACCENT, outline="")

        # Mode
        mode_frame = tk.Frame(right, bg=BG_MID, pady=10)
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(mode_frame, text="MODE DE DÉTECTION",
                 bg=BG_MID, fg=TEXT_SEC,
                 font=("Courier", 9)).pack(pady=(0, 6))
        btn_row = tk.Frame(mode_frame, bg=BG_MID)
        btn_row.pack()
        self.btn_alpha = tk.Button(btn_row, text="Alphabet",
                                   bg=ACCENT, fg=BG_DARK,
                                   font=("Courier", 10, "bold"),
                                   relief=tk.FLAT, padx=14, pady=4,
                                   cursor="hand2",
                                   command=lambda: self._set_mode("alphabet"))
        self.btn_alpha.pack(side=tk.LEFT, padx=4)
        self.btn_mots = tk.Button(btn_row, text="Mots",
                                  bg=BG_PANEL, fg=TEXT_SEC,
                                  font=("Courier", 10),
                                  relief=tk.FLAT, padx=14, pady=4,
                                  cursor="hand2",
                                  command=lambda: self._set_mode("mots"))
        self.btn_mots.pack(side=tk.LEFT, padx=4)

        # Historique
        hist_frame = tk.Frame(right, bg=BG_MID)
        hist_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        tk.Label(hist_frame, text="HISTORIQUE",
                 bg=BG_MID, fg=TEXT_SEC,
                 font=("Courier", 9)).pack(anchor=tk.W, padx=14, pady=(10, 4))
        self.hist_text = tk.Text(hist_frame, bg=BG_PANEL, fg=TEXT_PRI,
                                 font=("Courier", 16), relief=tk.FLAT,
                                 state=tk.DISABLED, wrap=tk.WORD,
                                 padx=10, pady=10)
        self.hist_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))
        tk.Button(hist_frame, text="Effacer",
                  bg=BG_PANEL, fg=TEXT_SEC,
                  font=("Courier", 9), relief=tk.FLAT,
                  cursor="hand2",
                  command=self._clear_history).pack(anchor=tk.E, padx=14, pady=(0, 8))

        # Toolbar
        toolbar = tk.Frame(self.root, bg=BG_MID, height=56)
        toolbar.pack(fill=tk.X, side=tk.BOTTOM)
        toolbar.pack_propagate(False)
        self.btn_start = tk.Button(toolbar, text="▶  Démarrer",
                                   bg=ACCENT, fg=BG_DARK,
                                   font=("Courier", 11, "bold"),
                                   relief=tk.FLAT, padx=20,
                                   cursor="hand2",
                                   command=self.start)
        self.btn_start.pack(side=tk.LEFT, padx=12, pady=10)
        self.btn_stop = tk.Button(toolbar, text="⏹  Stop",
                                  bg=BG_PANEL, fg=DANGER,
                                  font=("Courier", 11),
                                  relief=tk.FLAT, padx=16,
                                  cursor="hand2",
                                  state=tk.DISABLED,
                                  command=self.stop)
        self.btn_stop.pack(side=tk.LEFT, padx=4, pady=10)
        tk.Button(toolbar, text="🔊  Lire",
                  bg=BG_PANEL, fg=TEXT_SEC,
                  font=("Courier", 11),
                  relief=tk.FLAT, padx=16,
                  cursor="hand2",
                  command=self._speak).pack(side=tk.LEFT, padx=4, pady=10)
        tk.Button(toolbar, text="💾  Exporter",
                  bg=BG_PANEL, fg=TEXT_SEC,
                  font=("Courier", 11),
                  relief=tk.FLAT, padx=16,
                  cursor="hand2",
                  command=self._export).pack(side=tk.LEFT, padx=4, pady=10)
        self.label_status = tk.Label(toolbar, text="● Arrêté",
                                     bg=BG_MID, fg=DANGER,
                                     font=("Courier", 10))
        self.label_status.pack(side=tk.RIGHT, padx=20)

    def _set_mode(self, mode):
        self.mode.set(mode)
        if mode == "alphabet":
            self.btn_alpha.config(bg=ACCENT, fg=BG_DARK, font=("Courier", 10, "bold"))
            self.btn_mots.config(bg=BG_PANEL, fg=TEXT_SEC, font=("Courier", 10))
        else:
            self.btn_mots.config(bg=ACCENT2, fg=BG_DARK, font=("Courier", 10, "bold"))
            self.btn_alpha.config(bg=BG_PANEL, fg=TEXT_SEC, font=("Courier", 10))

    def _extract_landmarks(self, hand_landmarks) -> np.ndarray:
        wrist = hand_landmarks[0]
        coords = []
        for lm in hand_landmarks:
            coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return np.array(coords)

    def _update_conf_bar(self, conf: float):
        self.conf_bar_bg.update_idletasks()
        w = self.conf_bar_bg.winfo_width()
        fill_w = int(w * conf)
        color = ACCENT if conf > 0.8 else WARNING if conf > 0.5 else DANGER
        self.conf_bar_bg.coords(self.conf_bar, 0, 0, fill_w, 6)
        self.conf_bar_bg.itemconfig(self.conf_bar, fill=color)

    def _add_to_history(self, label: str):
        self.history.append(label)
        self.hist_text.config(state=tk.NORMAL)
        self.hist_text.delete(1.0, tk.END)
        self.hist_text.insert(tk.END, "  ".join(self.history[-50:]))
        self.hist_text.config(state=tk.DISABLED)
        self.hist_text.see(tk.END)

    def _clear_history(self):
        self.history = []
        self.hist_text.config(state=tk.NORMAL)
        self.hist_text.delete(1.0, tk.END)
        self.hist_text.config(state=tk.DISABLED)

    def _speak(self):
        try:
            import pyttsx3
            text = " ".join(self.history)
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS non disponible : {e}")

    def _export(self):
        os.makedirs("exports", exist_ok=True)
        path = f"exports/historique_{int(time.time())}.txt"
        with open(path, 'w') as f:
            f.write(" ".join(self.history))
        print(f"Historique exporté : {path}")

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.label_status.config(text="● En cours", fg=ACCENT)
        self.canvas.delete("placeholder")
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.label_status.config(text="● Arrêté", fg=DANGER)
        self.label_detected.config(text="—")
        self.label_conf.config(text="0%")
        self._update_conf_bar(0)

    def _capture_loop(self):
        last_label = ""
        last_time  = 0
        COOLDOWN   = 1.0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_img)

            label, conf = "", 0.0

            if result.hand_landmarks:
                hand   = result.hand_landmarks[0]
                coords = self._extract_landmarks(hand)

                h, w, _ = frame.shape
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 136), -1)

                if self.mode.get() == "alphabet" and self.alpha_ok:
                    pred_idx = self.alpha_model.predict([coords])[0]
                    conf     = self.alpha_model.predict_proba([coords])[0].max()
                    label    = self.alpha_encoder.inverse_transform([pred_idx])[0]

                elif self.mode.get() == "mots" and self.lstm_ok:
                    self.sequence.append(coords)
                    if len(self.sequence) == SEQUENCE_LENGTH:
                        X_pred = np.expand_dims(np.array(self.sequence), axis=0)
                        pred   = self.lstm_model.predict(X_pred, verbose=0)[0]
                        idx    = np.argmax(pred)
                        conf   = float(pred[idx])
                        label  = self.lstm_encoder.inverse_transform([idx])[0]

            if label and conf > 0.75:
                now = time.time()
                if label != last_label or (now - last_time) > COOLDOWN:
                    self.root.after(0, self._add_to_history, label)
                    last_label = label
                    last_time  = now

            self.root.after(0, self._update_ui, label, conf, frame)

    def _update_ui(self, label, conf, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((620, 465))
        photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

        if label:
            self.label_detected.config(text=label)
            self.label_conf.config(text=f"{conf*100:.0f}%")
            self._update_conf_bar(conf)
        else:
            self.label_detected.config(text="—")
            self.label_conf.config(text="0%")
            self._update_conf_bar(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = SignumApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()
