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

if __name__ == "__main__":
    unittest.main()
