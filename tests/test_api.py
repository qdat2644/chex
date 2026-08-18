from __future__ import annotations

import concurrent.futures
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from fastapi.testclient import TestClient
from PIL import Image

os.environ["CHEXPERT_AUTO_DOWNLOAD"] = "false"
os.environ["APP_ENV"] = "development"

from app.main import (
    _RATE_LIMIT_STORE,
    anonymize_filename,
    create_app,
    decode_image_or_dicom,
    lifespan,
    mask_phi,
    resolve_checkpoint_path,
    _download_hf_file,
)
from app.model import CheXpertPredictor, Heatmap, Prediction


class ApiTest(unittest.TestCase):
    def setUp(self):
        _RATE_LIMIT_STORE.clear()
        # Create Mock Predictor with a tiny neural net
        self.mock_predictor = CheXpertPredictor(None)
        self.mock_predictor.labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Pleural Effusion",
        ]
        self.mock_predictor.thresholds = {lbl: 0.5 for lbl in self.mock_predictor.labels}
        self.mock_predictor.model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(3, 5),
        )
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

        # Use app factory to inject mock predictor into app state
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
        # Ensure raw filename is masked with scan_<hash>.png
        self.assertNotIn("raw_patient_xray.jpg", payload["filename"])
        self.assertTrue(payload["filename"].startswith("scan_"))

    def test_anonymized_filename_uniqueness(self) -> None:
        name_1 = anonymize_filename("Patient_Alice_20260818.dcm")
        name_2 = anonymize_filename("Patient_Bob_20260818.dcm")
        self.assertNotEqual(name_1, name_2)
        self.assertTrue(name_1.startswith("scan_"))
        self.assertTrue(name_2.startswith("scan_"))
        self.assertTrue(name_1.endswith(".png"))
        self.assertTrue(name_2.endswith(".png"))

    def test_app_state_isolation(self) -> None:
        pred_a = CheXpertPredictor(None)
        pred_a.labels = ["Atelectasis"]
        pred_a.model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(3, 1),
        )
        pred_a.predict = MagicMock(return_value=[
            Prediction(label="Atelectasis", probability=0.99, positive=True, threshold=0.5, suspicion_level="High suspicion")
        ])
        pred_a.explain_finding = MagicMock(return_value=Heatmap(label="Atelectasis", probability=0.99, image_data_url="data:image/png;base64,dummy"))

        pred_b = CheXpertPredictor(None)
        pred_b.labels = ["Cardiomegaly"]
        pred_b.model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(3, 1),
        )
        pred_b.predict = MagicMock(return_value=[
            Prediction(label="Cardiomegaly", probability=0.11, positive=False, threshold=0.5, suspicion_level="Low suspicion")
        ])
        pred_b.explain_finding = MagicMock(return_value=Heatmap(label="Cardiomegaly", probability=0.11, image_data_url="data:image/png;base64,dummy"))

        app_a = create_app(pred_a)
        app_b = create_app(pred_b)

        client_a = TestClient(app_a)
        client_b = TestClient(app_b)

        img_bytes = io.BytesIO()
        Image.new("RGB", (32, 32), color="gray").save(img_bytes, format="PNG")
        raw_data = img_bytes.getvalue()

        res_a = client_a.post("/api/predict?include_heatmap=false", files={"file": ("test.png", raw_data, "image/png")})
        res_b = client_b.post("/api/predict?include_heatmap=false", files={"file": ("test.png", raw_data, "image/png")})

        self.assertEqual(res_a.json()["findings"][0]["label"], "Atelectasis")
        self.assertEqual(res_a.json()["findings"][0]["probability"], 0.99)

        self.assertEqual(res_b.json()["findings"][0]["label"], "Cardiomegaly")
        self.assertEqual(res_b.json()["findings"][0]["probability"], 0.11)

    def test_dicom_multiframe_rejection_before_decode(self) -> None:
        try:
            import pydicom
            from pydicom.dataset import Dataset, FileMetaDataset
            from pydicom.uid import ExplicitVRLittleEndian

            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
            meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
            meta.TransferSyntaxUID = ExplicitVRLittleEndian

            ds = Dataset()
            ds.file_meta = meta
            ds.is_little_endian = True
            ds.is_implicit_VR = False
            ds.NumberOfFrames = 5
            ds.Rows = 100
            ds.Columns = 100
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.PixelData = b"\x00" * (5 * 100 * 100 * 2)

            dcm_buf = io.BytesIO()
            pydicom.dcmwrite(dcm_buf, ds, write_like_original=False)
            dcm_buf.seek(0)

            response = self.client.post(
                "/api/predict",
                files={"file": ("multiframe_study.dcm", dcm_buf.getvalue(), "application/dicom")},
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("Multi-frame DICOM series are not supported", response.json()["detail"])
        except ImportError:
            pass

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
        self.assertEqual(payload["version"], "1.0.0")
        self.assertIn("revision", payload)
        self.assertIn("checkpoint_sha256", payload)
        self.assertIn("thresholds_sha256", payload)

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

        self.assertNotIn("test_patient_name_001.jpg", payload["filename"])
        self.assertTrue(payload["filename"].startswith("scan_"))

        explain_res = self.client.post(
            "/api/explain",
            files={"file": ("test.jpg", image_bytes.getvalue(), "image/jpeg")},
            data={"label": "Cardiomegaly"},
        )
        self.assertEqual(explain_res.status_code, 200)
        self.assertEqual(explain_res.json()["label"], "Cardiomegaly")

    def test_authentication_protection(self) -> None:
        """
        Task 5: Test API Key authentication on protected /api/* endpoints.
        """
        secure_app = create_app(self.mock_predictor, api_key_override="test_secret_api_key_2026")
        secure_client = TestClient(secure_app)

        # 1. /health is public without auth
        health_res = secure_client.get("/health")
        self.assertEqual(health_res.status_code, 200)

        # 2. /api/model-info without auth returns 401
        unauth_res = secure_client.get("/api/model-info")
        self.assertEqual(unauth_res.status_code, 401)
        self.assertIn("Unauthorized", unauth_res.json()["detail"])

        # 3. /api/model-info with wrong token returns 401
        wrong_token_res = secure_client.get("/api/model-info", headers={"Authorization": "Bearer wrong_token"})
        self.assertEqual(wrong_token_res.status_code, 401)

        # 4. /api/model-info with valid Bearer token returns 200
        auth_res = secure_client.get("/api/model-info", headers={"Authorization": "Bearer test_secret_api_key_2026"})
        self.assertEqual(auth_res.status_code, 200)

        # 5. /api/model-info with X-API-Key header returns 200
        x_key_res = secure_client.get("/api/model-info", headers={"X-API-Key": "test_secret_api_key_2026"})
        self.assertEqual(x_key_res.status_code, 200)

    def test_rate_limiting_trigger(self) -> None:
        """
        Task 5: Test rate limiter returns 429 when threshold exceeded.
        """
        limited_app = create_app(self.mock_predictor, rate_limit_override=3)
        limited_client = TestClient(limited_app)

        # Make 3 requests (within limit)
        for _ in range(3):
            r = limited_client.get("/api/model-info")
            self.assertEqual(r.status_code, 200)

        # 4th request must be rate-limited with 429
        rate_limited_res = limited_client.get("/api/model-info")
        self.assertEqual(rate_limited_res.status_code, 429)
        self.assertEqual(rate_limited_res.headers.get("Retry-After"), "60")
        self.assertIn("Rate limit exceeded", rate_limited_res.json()["detail"])

    def test_production_missing_secret_fails_startup(self) -> None:
        """
        Task 1 & Task 2: Production mode missing secret halts startup.
        """
        with patch.dict(os.environ, {"APP_ENV": "production", "PHI_HMAC_SECRET": ""}):
            app_prod = create_app()
            with self.assertRaises(RuntimeError) as ctx:
                with TestClient(app_prod):
                    pass
            self.assertIn("CRITICAL", str(ctx.exception))

    def test_production_checksum_mismatch_fails_startup(self) -> None:
        """
        Task 2: Production mode with corrupt/mismatched checkpoint hash halts startup.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_ckpt = Path(tmpdir) / "corrupt_model.pt"
            fake_ckpt.write_text("corrupted_data_bytes", encoding="utf-8")
            with patch.dict(os.environ, {
                "APP_ENV": "production",
                "PHI_HMAC_SECRET": "a" * 32,
                "CHEXPERT_CHECKPOINT": str(fake_ckpt),
                "CHEXPERT_CHECKPOINT_SHA256": "0000000000000000000000000000000000000000000000000000000000000000",
            }):
                app_prod = create_app()
                with self.assertRaises(RuntimeError) as ctx:
                    with TestClient(app_prod):
                        pass
                self.assertIn("Integrity check failed", str(ctx.exception))

    def test_batch_predict_chunking_and_client_ids(self) -> None:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "test_model.pt"
            success = _download_hf_file("http://localhost/dummy", target)
            self.assertFalse(success)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
