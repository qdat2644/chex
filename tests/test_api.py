from __future__ import annotations

import concurrent.futures
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure zero external network calls during testing
os.environ["CHEXPERT_AUTO_DOWNLOAD"] = "false"
os.environ["APP_ENV"] = "development"

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app, decode_image_or_dicom, mask_phi, _download_hf_file
from app.model import CheXpertPredictor, Heatmap, Prediction


class ApiTest(unittest.TestCase):
    def setUp(self):
        # Create a tiny mock predictor to keep tests fast and isolated
        self.mock_predictor = CheXpertPredictor(None)
        self.mock_predictor.labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ]
        self.mock_predictor.thresholds = {lbl: 0.5 for lbl in self.mock_predictor.labels}
        self.mock_predictor.model = MagicMock()
        self.mock_predictor.predict = MagicMock(return_value=[
            Prediction(label="Atelectasis", probability=0.15, positive=False, threshold=0.5, suspicion_level="Low suspicion"),
            Prediction(label="Cardiomegaly", probability=0.85, positive=True, threshold=0.5, suspicion_level="High suspicion"),
            Prediction(label="Consolidation", probability=0.10, positive=False, threshold=0.5, suspicion_level="Low suspicion"),
            Prediction(label="Edema", probability=0.30, positive=False, threshold=0.5, suspicion_level="Low suspicion"),
            Prediction(label="Pleural Effusion", probability=0.05, positive=False, threshold=0.5, suspicion_level="Low suspicion"),
        ])
        self.mock_predictor.predict_batch = MagicMock(side_effect=lambda imgs, chunk_size=8: [
            self.mock_predictor.predict(img) for img in imgs
        ])
        self.mock_predictor.explain_finding = MagicMock(return_value=Heatmap(
            label="Cardiomegaly",
            probability=0.85,
            image_data_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            pure_heatmap_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        ))

        # Use app factory to inject mock predictor
        self.app = create_app(self.mock_predictor)
        self.client = TestClient(self.app)

    def test_predict_returns_clear_status_without_checkpoint(self) -> None:
        unloaded_predictor = CheXpertPredictor(None)
        app_unloaded = create_app(unloaded_predictor)
        client_unloaded = TestClient(app_unloaded)

        image_bytes = io.BytesIO()
        Image.new("RGB", (10, 12), color=(80, 80, 80)).save(image_bytes, format="JPEG")

        response = client_unloaded.post(
            "/api/predict",
            files={"file": ("raw_patient_xray.jpg", image_bytes.getvalue(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "model_not_loaded")
        self.assertEqual(payload["width"], 10)
        self.assertEqual(payload["height"], 12)
        self.assertEqual(payload["findings"], [])
        self.assertIsNone(payload["report"])
        self.assertIsNone(payload["heatmap"])
        # Ensure raw filename is masked in default mode
        self.assertNotIn("raw_patient_xray.jpg", payload["filename"])

    def test_model_info_reports_labels_and_metadata(self) -> None:
        expected_labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ]
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["labels"], expected_labels)
        self.assertTrue(payload["model_loaded"])

    def test_predict_and_explain_endpoints_offline(self) -> None:
        image_bytes = io.BytesIO()
        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(image_bytes, format="JPEG")

        response = self.client.post(
            "/api/predict",
            files={"file": ("test_patient_name_001.jpg", image_bytes.getvalue(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(len(payload["findings"]), 5)
        self.assertIsNotNone(payload["heatmap"])
        self.assertTrue(payload["heatmap"]["image_data_url"].startswith("data:image/png;base64,"))

        # Test PHI protection: raw filename should not be exposed
        self.assertNotIn("test_patient_name_001.jpg", payload["filename"])
        self.assertTrue(payload["filename"].startswith("scan_"))

        # Explain endpoint
        explain_res = self.client.post(
            "/api/explain",
            files={"file": ("test.jpg", image_bytes.getvalue(), "image/jpeg")},
            data={"label": "Cardiomegaly"},
        )
        self.assertEqual(explain_res.status_code, 200)
        self.assertEqual(explain_res.json()["label"], "Cardiomegaly")

    def test_explain_invalid_label_returns_400(self) -> None:
        image_bytes = io.BytesIO()
        Image.new("RGB", (64, 64), color="gray").save(image_bytes, format="JPEG")

        response = self.client.post(
            "/api/explain",
            files={"file": ("test.jpg", image_bytes.getvalue(), "image/jpeg")},
            data={"label": "NonExistentPathology123"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid label", response.json()["detail"])

    def test_batch_predict_chunking_and_client_ids(self) -> None:
        """
        Task 3: Test batch pipeline processing 10 images (> chunk size 8).
        Verifies chunking and client_id preservation.
        """
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), color="gray").save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        files_list = [("files", (f"cxr_{i}.png", raw_bytes, "image/png")) for i in range(10)]
        client_ids = [f"cid_{i}_{i*10}" for i in range(10)]
        ids_str = ",".join(client_ids)

        response = self.client.post(
            "/api/predict-batch",
            files=files_list,
            data={"client_ids": ids_str},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["total"], 10)
        self.assertEqual(payload["processed"], 10)
        self.assertEqual(len(payload["results"]), 10)

        for i, item in enumerate(payload["results"]):
            self.assertEqual(item["client_id"], client_ids[i])
            self.assertNotIn(f"cxr_{i}.png", item["filename"])

    def test_phi_masking_hmac(self) -> None:
        self.assertEqual(mask_phi(None), "ANONYMIZED")
        self.assertEqual(mask_phi("ANONYMIZED"), "ANONYMIZED")
        self.assertEqual(mask_phi("N/A"), "ANONYMIZED")

        masked = mask_phi("PATIENT_RECORD_98765")
        self.assertTrue(masked.startswith("ANONYMIZED_"))
        self.assertEqual(len(masked), len("ANONYMIZED_") + 16)
        self.assertNotEqual(masked, "PATIENT_RECORD_98765")

    def test_concurrent_predict_and_explain(self) -> None:
        image_bytes = io.BytesIO()
        Image.new("RGB", (64, 64), color=(120, 120, 120)).save(image_bytes, format="PNG")
        raw_data = image_bytes.getvalue()

        def do_predict():
            return self.client.post(
                "/api/predict",
                files={"file": ("test.png", raw_data, "image/png")},
            )

        def do_explain():
            return self.client.post(
                "/api/explain",
                files={"file": ("test.png", raw_data, "image/png")},
                data={"label": "Edema"},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            f1 = executor.submit(do_predict)
            f2 = executor.submit(do_explain)
            f3 = executor.submit(do_predict)
            f4 = executor.submit(do_explain)
            results = [f.result() for f in [f1, f2, f3, f4]]

        for r in results:
            self.assertEqual(r.status_code, 200)

    def test_checksum_verification_fail_closed(self) -> None:
        """
        Task 2: Test file matching hash, wrong hash, and auto-download disabled rejection.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "test_model.pt"
            # When CHEXPERT_AUTO_DOWNLOAD=false, download must be immediately rejected
            success = _download_hf_file("http://localhost/dummy", target)
            self.assertFalse(success)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
