"""
tts.py — Synthèse vocale pour Signum HandSpeak
Mois 4 : lire l'historique des signes à voix haute

Usage :
    from src.tts import speak, speak_history
"""

import threading
import queue
import time

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("pyttsx3 non disponible — installe avec : pip install pyttsx3")


class TTSEngine:
    """
    Moteur de synthèse vocale asynchrone.
    Les textes sont mis en file d'attente et lus sans bloquer l'interface.
    """

    def __init__(self):
        self.queue   = queue.Queue()
        self.running = False
        self.engine  = None
        self._init_engine()

    def _init_engine(self):
        """Initialise pyttsx3 avec les bons paramètres."""
        if not TTS_AVAILABLE:
            return
        try:
            self.engine = pyttsx3.init()

            # Vitesse de parole (mots/minute) — 150 = naturel
            self.engine.setProperty('rate', 150)

            # Volume (0.0 → 1.0)
            self.engine.setProperty('volume', 1.0)

            # Cherche une voix française si disponible
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'french' in voice.name.lower() or 'fr' in voice.id.lower():
                    self.engine.setProperty('voice', voice.id)
                    print(f"Voix française trouvée : {voice.name}")
                    break

            print("Moteur TTS initialisé")
        except Exception as e:
            print(f"Erreur TTS : {e}")
            self.engine = None

    def start(self):
        """Démarre le thread de lecture."""
        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()

    def stop(self):
        """Arrête le thread de lecture."""
        self.running = False
        self.queue.put(None)  # signal d'arrêt

    def say(self, text: str):
        """Ajoute un texte à la file de lecture."""
        if text.strip():
            self.queue.put(text)

    def say_now(self, text: str):
        """Lit un texte immédiatement (bloque jusqu'à la fin)."""
        if not self.engine or not TTS_AVAILABLE:
            print(f"[TTS] {text}")
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Erreur TTS : {e}")

    def _worker(self):
        """Thread de lecture asynchrone."""
        while self.running:
            try:
                text = self.queue.get(timeout=0.5)
                if text is None:
                    break
                self.say_now(text)
                self.queue.task_done()
            except queue.Empty:
                continue


# ── Instance globale ─────────────────────────────────────────────────────────
_engine = TTSEngine()


def speak(text: str):
    """Lit un texte à voix haute (non bloquant)."""
    _engine.say(text)


def speak_now(text: str):
    """Lit un texte immédiatement (bloquant)."""
    _engine.say_now(text)


def speak_history(history: list):
    """Lit tout l'historique des signes."""
    if not history:
        speak_now("L'historique est vide.")
        return
    text = " ".join(history)
    speak_now(text)


def speak_letter(letter: str, confidence: float):
    """Lit une lettre détectée si la confiance est suffisante."""
    if confidence > 0.85:
        speak(letter)


def set_rate(rate: int):
    """Change la vitesse de parole (100-200 recommandé)."""
    if _engine.engine:
        _engine.engine.setProperty('rate', rate)


def set_volume(volume: float):
    """Change le volume (0.0 → 1.0)."""
    if _engine.engine:
        _engine.engine.setProperty('volume', max(0.0, min(1.0, volume)))


def start():
    """Démarre le moteur TTS."""
    _engine.start()


def stop():
    """Arrête le moteur TTS."""
    _engine.stop()


if __name__ == "__main__":
    print("=== Test TTS — Signum HandSpeak ===")
    start()

    print("Test 1 : lettre")
    speak_now("A")
    time.sleep(0.5)

    print("Test 2 : mot")
    speak_now("Bonjour")
    time.sleep(0.5)

    print("Test 3 : phrase")
    speak_now("Signum HandSpeak, traducteur de langue des signes.")
    time.sleep(1)

    print("Test 4 : historique")
    speak_history(["B", "O", "N", "J", "O", "U", "R"])
    time.sleep(2)

    stop()
    print("TTS terminé.")
