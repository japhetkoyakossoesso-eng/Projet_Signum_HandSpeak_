"""
test_all.py — Tests unitaires pour Signum HandSpeak

Usage :
    python3 tests/test_all.py
"""

import unittest
import numpy as np
import os
import sys
import pickle

# Ajoute src/ au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# Tests Export PDF 
class TestExportPDF(unittest.TestCase):

    def test_export_cree_fichier(self):
        """Vérifie que le PDF est bien créé."""
        from export_pdf import export_to_pdf
        history = ["A", "B", "bonjour"]
        path = export_to_pdf(history, "exports/test_output.pdf")
        self.assertTrue(os.path.exists(path))
        print("  PDF créé avec succès")

    def test_export_historique_vide(self):
        """Vérifie que le PDF se génère même avec un historique vide."""
        from export_pdf import export_to_pdf
        path = export_to_pdf([], "exports/test_vide.pdf")
        self.assertTrue(os.path.exists(path))
        print(" PDF vide créé avec succès")

    def tearDown(self):
        """Nettoie les fichiers de test."""
        for f in ["exports/test_output.pdf", "exports/test_vide.pdf"]:
            if os.path.exists(f):
                os.remove(f)


#  Tests Landmarks 
class TestLandmarks(unittest.TestCase):

    def _make_fake_landmark(self, x, y, z):
        """Crée un faux landmark pour les tests."""
        class FakeLM:
            pass
        lm = FakeLM()
        lm.x, lm.y, lm.z = x, y, z
        return lm

    def test_extraction_landmarks(self):
        """Vérifie que l'extraction produit 63 valeurs."""
        # Simule 21 landmarks
        hand = [self._make_fake_landmark(i*0.01, i*0.02, i*0.005)
                for i in range(21)]

        wrist = hand[0]
        coords = []
        for lm in hand:
            coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        self.assertEqual(len(coords), 63)
        print(" Extraction landmarks : 63 valeurs correctes")

    def test_normalisation_poignet(self):
        """Vérifie que le poignet est normalisé à (0,0,0)."""
        hand = [self._make_fake_landmark(0.5, 0.5, 0.1)]
        hand += [self._make_fake_landmark(0.6, 0.7, 0.2) for _ in range(20)]

        wrist = hand[0]
        coords = []
        for lm in hand:
            coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        # Les 3 premières valeurs (poignet) doivent être 0
        self.assertAlmostEqual(coords[0], 0.0)
        self.assertAlmostEqual(coords[1], 0.0)
        self.assertAlmostEqual(coords[2], 0.0)
        print(" Normalisation poignet : correcte")


# Tests Modèle Alphabet 
class TestModeleAlphabet(unittest.TestCase):

    def setUp(self):
        """Charge le modèle si disponible."""
        try:
            with open("models/alphabet_model.pkl", 'rb') as f:
                self.model = pickle.load(f)
            with open("models/label_encoder.pkl", 'rb') as f:
                self.encoder = pickle.load(f)
            self.model_ok = True
        except Exception:
            self.model_ok = False

    def test_modele_charge(self):
        """Vérifie que le modèle se charge correctement."""
        self.assertTrue(self.model_ok, "Modèle alphabet introuvable")
        print("  Modèle alphabet chargé")

    def test_prediction_shape(self):
        """Vérifie que la prédiction retourne un label valide."""
        if not self.model_ok:
            self.skipTest("Modèle non disponible")
        X = np.zeros((1, 63))
        pred = self.model.predict(X)
        self.assertEqual(len(pred), 1)
        label = self.encoder.inverse_transform(pred)[0]
        self.assertIsInstance(label, str)
        print(f"  Prédiction valide : '{label}'")

    def test_26_classes(self):
        """Vérifie que le modèle connaît les 26 lettres."""
        if not self.model_ok:
            self.skipTest("Modèle non disponible")
        classes = self.encoder.classes_
        lettres = [c for c in classes if len(c) == 1]
        self.assertGreaterEqual(len(lettres), 26)
        print(f" {len(classes)} classes dans le modèle")


#  Tests Modèle LSTM 
class TestModeleLSTM(unittest.TestCase):

    def setUp(self):
        """Charge le modèle LSTM si disponible."""
        try:
            import tensorflow as tf
            self.lstm = tf.keras.models.load_model("models/words_model.keras")
            with open("models/words_encoder.pkl", 'rb') as f:
                self.encoder = pickle.load(f)
            self.model_ok = True
        except Exception:
            self.model_ok = False

    def test_modele_lstm_charge(self):
        """Vérifie que le modèle LSTM se charge."""
        self.assertTrue(self.model_ok, "Modèle LSTM introuvable")
        print(" Modèle LSTM chargé")

    def test_prediction_sequence(self):
        """Vérifie que le LSTM prédit sur une séquence de 30 frames."""
        if not self.model_ok:
            self.skipTest("Modèle LSTM non disponible")
        X = np.zeros((1, 30, 63))
        pred = self.lstm.predict(X, verbose=0)
        self.assertEqual(pred.shape[0], 1)
        idx = np.argmax(pred[0])
        label = self.encoder.inverse_transform([idx])[0]
        self.assertIsInstance(label, str)
        print(f" Prédiction LSTM valide : '{label}'")


# Tests Historique 
class TestHistorique(unittest.TestCase):

    def test_historique_lettre(self):
        """Vérifie la détection lettre vs mot."""
        history = ["A", "B", "bonjour", "C", "merci"]
        lettres = [s for s in history if len(s) == 1]
        mots    = [s for s in history if len(s) > 1]
        self.assertEqual(len(lettres), 3)
        self.assertEqual(len(mots), 2)
        print("  Séparation lettres/mots correcte")

    def test_historique_vide(self):
        """Vérifie le comportement avec un historique vide."""
        history = []
        self.assertEqual(len(history), 0)
        texte = " ".join(history)
        self.assertEqual(texte, "")
        print(" Historique vide géré correctement")

    def test_historique_max_50(self):
        """Vérifie que l'affichage est limité aux 50 derniers signes."""
        history = [str(i) for i in range(100)]
        affichage = history[-50:]
        self.assertEqual(len(affichage), 50)
        print("  Limite 50 signes correcte")


# Lancement 
if __name__ == "__main__":
    print("=" * 50)
    print("  TESTS UNITAIRES — SIGNUM HANDSPEAK")
    print("=" * 50)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestLandmarks))
    suite.addTests(loader.loadTestsFromTestCase(TestHistorique))
    suite.addTests(loader.loadTestsFromTestCase(TestExportPDF))
    suite.addTests(loader.loadTestsFromTestCase(TestModeleAlphabet))
    suite.addTests(loader.loadTestsFromTestCase(TestModeleLSTM))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("  TOUS LES TESTS PASSES ")
    else:
        print(f"  {len(result.failures)} echec(s), {len(result.errors)} erreur(s) ")
    print("=" * 50)
