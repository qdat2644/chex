from __future__ import annotations

import io
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.model import CheXpertPredictor
from app.main import app


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

    def test_model_info_reports_v2_checkpoint_and_label_order(self) -> None:
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
        self.assertTrue(payload["checkpoint"].endswith("checkpoints\\chexpert_densenet121_v2.pt") or payload["checkpoint"].endswith("checkpoints/chexpert_densenet121_v2.pt"))
        self.assertTrue(payload["thresholds_loaded"])
        self.assertEqual(list(payload["thresholds"].keys()), expected_labels)

    def test_predict_returns_report_and_heatmap_with_checkpoint(self) -> None:
        checkpoint = Path("outputs/chex_smoke.pt")
        sample = Path("archive/train/patient00001/study1/view1_frontal.jpg")
        if not checkpoint.exists() or not sample.exists():
            self.skipTest("Smoke checkpoint or sample image is missing.")

        original_predictor = main_module.predictor
        main_module.predictor = CheXpertPredictor(checkpoint)
        try:
            client = TestClient(app)
            with sample.open("rb") as image_file:
                response = client.post(
                    "/api/predict",
                    files={"file": ("xray.jpg", image_file, "image/jpeg")},
                )
        finally:
            main_module.predictor = original_predictor

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["findings"]), 5)
        self.assertIn("threshold", payload["findings"][0])
        self.assertIsInstance(payload["report"], str)
        self.assertTrue(payload["heatmap"]["image_data_url"].startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
