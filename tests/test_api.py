from __future__ import annotations

import concurrent.futures
import io
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.main import app, mask_phi, _download_hf_file
from app.model import CheXpertPredictor


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict_returns_clear_status_without_checkpoint(self) -> None:
        image_bytes = io.BytesIO()
        Image.new("RGB", (10, 12), color=(80, 80, 80)).save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        original_predictor = main_module.predictor
        main_module.predictor = CheXpertPredictor(None)
        try:
            response = self.client.post(
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

        response = self.client.get("/api/model-info")
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

        response = self.client.post(
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
        explain_res = self.client.post(
            "/api/explain",
            files={"file": ("test_cxr.jpg", image_bytes.getvalue(), "image/jpeg")},
            data={"label": "Cardiomegaly"},
        )
        self.assertEqual(explain_res.status_code, 200)
        explain_payload = explain_res.json()
        self.assertEqual(explain_payload["label"], "Cardiomegaly")
        self.assertTrue(explain_payload["image_data_url"].startswith("data:image/png;base64,"))
        self.assertTrue(explain_payload["pure_heatmap_url"].startswith("data:image/png;base64,"))

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

    def test_batch_predict_endpoint_with_client_ids(self) -> None:
        buf1 = io.BytesIO()
        Image.new("RGB", (64, 64), color="gray").save(buf1, format="PNG")
        buf2 = io.BytesIO()
        Image.new("RGB", (64, 64), color="black").save(buf2, format="PNG")

        custom_id_1 = "custom_client_uuid_001"
        custom_id_2 = "custom_client_uuid_002"

        response = self.client.post(
            "/api/predict-batch",
            files=[
                ("files", ("duplicate_cxr.png", buf1.getvalue(), "image/png")),
                ("files", ("duplicate_cxr.png", buf2.getvalue(), "image/png")),
            ],
            data=[
                ("client_ids", custom_id_1),
                ("client_ids", custom_id_2),
            ]
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["processed"], 2)
        self.assertEqual(len(payload["results"]), 2)
        self.assertEqual(payload["results"][0]["client_id"], custom_id_1)
        self.assertEqual(payload["results"][1]["client_id"], custom_id_2)

    def test_phi_masking_hmac(self) -> None:
        self.assertEqual(mask_phi(None), "ANONYMIZED")
        self.assertEqual(mask_phi("ANONYMIZED"), "ANONYMIZED")
        self.assertEqual(mask_phi("N/A"), "ANONYMIZED")
        
        # Real patient name/ID should be masked with HMAC-SHA256 fingerprint of at least 16 chars
        masked = mask_phi("PATIENT_RECORD_98765")
        self.assertTrue(masked.startswith("ANONYMIZED_"))
        self.assertEqual(len(masked), len("ANONYMIZED_") + 16)
        self.assertNotEqual(masked, "PATIENT_RECORD_98765")

    def test_concurrent_predict_and_explain(self) -> None:
        """
        Verify thread safety when concurrent predict and explain requests are executed.
        """
        image_bytes = io.BytesIO()
        Image.new("RGB", (128, 128), color=(120, 120, 120)).save(image_bytes, format="PNG")
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

    def test_checksum_verification_rejection(self) -> None:
        """
        Verify that downloading with a mismatched SHA-256 hash gets rejected and cleaned up.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "test_model.pt"
            # Attempt to verify against wrong checksum
            fake_url = "https://raw.githubusercontent.com/qdat2644/chex/main/README.md"
            wrong_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
            success = _download_hf_file(fake_url, target, expected_sha256=wrong_sha256)
            self.assertFalse(success)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
