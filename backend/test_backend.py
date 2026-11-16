import unittest
from fastapi.testclient import TestClient
from app.main import app

class TestBackendAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "online")

    def test_get_datasets(self):
        response = self.client.get("/datasets")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertIn("name", data[0])

    def test_get_metrics(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("accuracy", data)
        self.assertIn("confusion_matrix", data)

    def test_predict_unified(self):
        response = self.client.post(
            "/predict", 
            data={"text": "I absolutely love this framework, it is amazing!"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["label"], "Positive")
        self.assertGreater(data["confidence"], 0.5)

    def test_predict_emotion(self):
        response = self.client.post(
            "/predict/emotion", 
            json={"text": "I am so happy and excited today!"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["emotion"], "Happiness")
        self.assertGreater(data["confidence"], 0.5)
        self.assertIn("probabilities", data)
        self.assertEqual(len(data["probabilities"]), 7)

    def test_openrouter_mock_sentiment(self):
        from unittest.mock import patch
        with patch("app.services.openrouter_service.OpenRouterService.is_configured", True):
            with patch("app.services.openrouter_service.OpenRouterService._call_openrouter") as mock_call:
                mock_call.return_value = '{"label": "Positive", "confidence": 0.98, "scores": {"positive": 0.98, "negative": 0.01, "neutral": 0.01}}'
                response = self.client.post("/predict", data={"text": "Outstanding service and experience!"})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["label"], "Positive")
                self.assertAlmostEqual(data["confidence"], 0.98, places=2)
                self.assertAlmostEqual(data["scores"]["positive"], 0.98, places=2)

    def test_openrouter_mock_emotion(self):
        from unittest.mock import patch
        with patch("app.services.openrouter_service.OpenRouterService.is_configured", True):
            with patch("app.services.openrouter_service.OpenRouterService._call_openrouter") as mock_call:
                mock_call.return_value = '{"emotion": "Happiness", "confidence": 0.96, "probabilities": {"Happiness": 0.96, "Sadness": 0.01, "Anger": 0.01, "Fear": 0.0, "Surprise": 0.01, "Disgust": 0.0, "Neutral": 0.01}}'
                response = self.client.post("/predict/emotion", json={"text": "I am thrilled and ecstatic!"})
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["emotion"], "Happiness")
                self.assertAlmostEqual(data["confidence"], 0.96, places=2)
                self.assertIn("Happiness", data["probabilities"])

if __name__ == "__main__":
    unittest.main()
