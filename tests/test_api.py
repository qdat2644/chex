from __future__ import annotations

import io
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.main import app
from app.model import CheXpertPredictor


class ApiTest(unittest.TestCase):
    def test_predict_returns_clear_status_without_checkpoint(self) -> None:
        image_bytes = io.BytesIO()
        Image.new("RGB", (10, 12), color=(80, 80, 80)).save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        original_predictor = main_module.predictor
        main_module.predictor = CheXpertPredictor(None)
        try:
            client = TestClient(app)
            response = client.post(
                "/api/predict",
                files={"file": ("xray.jpg", image_bytes, "image/jpeg")},
            )
        finally:
            main_module.predictor = original_predictor

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "model_not_loaded")
        self.assertEqual(payload["width"], 10)
        self.assertEqual(payload["height"], 12)
        self.assertEqual(payload["findings"], [])
        self.assertIsNone(payload["report"])
        self.assertIsNone(payload["heatmap"])

    def test_model_info_reports_checkpoint_and_label_order(self) -> None:
        expected_labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ]

        client = TestClient(app)
        response = client.get("/api/model-info")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["labels"], expected_labels)
        self.assertTrue(payload["checkpoint"].endswith(".pt"))
        self.assertTrue(payload["thresholds_loaded"])
        self.assertEqual(list(payload["thresholds"].keys()), expected_labels)

    def test_predict_and_explain_endpoints(self) -> None:
        image_bytes = io.BytesIO()
        Image.new("RGB", (224, 224), color=(100, 100, 100)).save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        client = TestClient(app)
        response = client.post(
            "/api/predict",
            files={"file": ("test_cxr.jpg", image_bytes.getvalue(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["findings"]), 5)
        self.assertIn("quality", payload)
        self.assertIn("dicom", payload)
        self.assertIsNotNone(payload["heatmap"])
        self.assertTrue(payload["heatmap"]["image_data_url"].startswith("data:image/png;base64,"))

        # Test on-demand /api/explain endpoint
        explain_res = client.post(
            "/api/explain",
            files={"file": ("test_cxr.jpg", image_bytes.getvalue(), "image/jpeg")},
            data={"label": "Cardiomegaly"},
        )
        self.assertEqual(explain_res.status_code, 200)
        explain_payload = explain_res.json()
        self.assertEqual(explain_payload["label"], "Cardiomegaly")
        self.assertTrue(explain_payload["image_data_url"].startswith("data:image/png;base64,"))
        self.assertTrue(explain_payload["pure_heatmap_url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
