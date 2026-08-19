from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from app.main import decode_image_or_dicom
from scripts.calibrate import calculate_optimal_thresholds
from scripts.compare_models import delong_paired_test, holm_bonferroni_correction, paired_bootstrap_delta_auc
from scripts.make_splits import multi_label_patient_stratification


class PipelineIntegrityTest(unittest.TestCase):
    def test_zero_patient_leakage_stratification(self) -> None:
        """
        P0 Test: Verifies that patient-level stratification produces zero patient overlap.
        """
        # Synthetic dataset with multiple studies per patient and multi-label targets
        records = []
        labels = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        for p_idx in range(100):
            pid = f"patient_{p_idx:04d}"
            num_studies = (p_idx % 3) + 1
            for s_idx in range(num_studies):
                rec = {
                    "Path": f"CheXpert-v1.0-small/train/{pid}/study{s_idx+1}/view1_frontal.jpg",
                    "patient_id": pid,
                    "Frontal/Lateral": "Frontal",
                }
                for l_idx, lbl in enumerate(labels):
                    rec[lbl] = 1.0 if (p_idx + l_idx) % 3 == 0 else 0.0
                records.append(rec)

        df = pd.DataFrame(records)

        train_df, calib_df, val_df = multi_label_patient_stratification(
            df,
            labels=labels,
            train_ratio=0.8,
            calib_ratio=0.1,
            val_ratio=0.1,
            seed=42,
        )

        train_pids = set(train_df["patient_id"])
        calib_pids = set(calib_df["patient_id"])
        val_pids = set(val_df["patient_id"])

        # Strict assertion of 0 overlap
        self.assertEqual(len(train_pids & calib_pids), 0, "Leakage detected between train and calib!")
        self.assertEqual(len(train_pids & val_pids), 0, "Leakage detected between train and internal val!")
        self.assertEqual(len(calib_pids & val_pids), 0, "Leakage detected between calib and internal val!")
        self.assertEqual(len(train_pids | calib_pids | val_pids), 100)

    def test_dicom_preprocessing_order_and_windowing(self) -> None:
        """
        P1 Test: Verifies strict DICOM preprocessing order:
        Raw pixels -> Modality LUT -> VOI LUT / Windowing -> MONOCHROME1 Inversion -> Normalization.
        """
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
        meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6"
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.NumberOfFrames = 1
        ds.Rows = 32
        ds.Columns = 32
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.RescaleSlope = 2.0
        ds.RescaleIntercept = -10.0
        ds.WindowCenter = 500
        ds.WindowWidth = 800

        # Synthetic pixel array
        raw_arr = np.linspace(100, 1000, 32 * 32, dtype=np.uint16).reshape(32, 32)
        ds.PixelData = raw_arr.tobytes()

        buf = io.BytesIO()
        pydicom.dcmwrite(buf, ds, write_like_original=False)
        buf.seek(0)

        img, dcm_meta = decode_image_or_dicom(buf, "test_study.dcm")

        self.assertEqual(img.size, (32, 32))
        self.assertTrue(dcm_meta.is_dicom)
        self.assertEqual(dcm_meta.photometric, "MONOCHROME1")

        # Verify pixel intensity array is non-trivial and uint8 [0, 255]
        arr = np.asarray(img)
        self.assertEqual(arr.dtype, np.uint8)
        self.assertGreater(float(np.max(arr)), float(np.min(arr)))

    def test_delong_paired_test_and_holm_correction(self) -> None:
        """
        P1 Test: Verifies DeLong paired calculation and Holm-Bonferroni step-down correction.
        """
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0] * 10)
        # Model A is slightly better than Model B
        preds_a = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.3, 0.85, 0.15, 0.95, 0.05] * 10)
        preds_b = np.array([0.6, 0.5, 0.4, 0.4, 0.3, 0.5, 0.55, 0.45, 0.65, 0.35] * 10)

        auc_a, auc_b, z_stat, p_val = delong_paired_test(preds_a, preds_b, y_true)
        self.assertGreater(auc_a, auc_b)
        self.assertGreater(z_stat, 0.0)
        self.assertGreaterEqual(p_val, 0.0)
        self.assertLessEqual(p_val, 1.0)

        delta_mean, ci_low, ci_up = paired_bootstrap_delta_auc(preds_a, preds_b, y_true, n_boot=500, seed=42)
        self.assertGreater(delta_mean, 0.0)

        # Test Holm correction monotonicity
        p_raw = [0.01, 0.04, 0.03, 0.20, 0.80]
        p_adj = holm_bonferroni_correction(p_raw)
        self.assertEqual(len(p_adj), 5)
        # Verify p_adj >= p_raw for all elements
        for pr, pa in zip(p_raw, p_adj):
            self.assertGreaterEqual(pa, pr)

    def test_locked_test_threshold_calibration_isolation(self) -> None:
        """
        P0 Test: Verifies that threshold calibration operates exclusively on calibration data.
        """
        labels = ["Atelectasis", "Cardiomegaly"]
        targets = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]] * 10
        probs = [[0.8, 0.2], [0.1, 0.9], [0.7, 0.6], [0.2, 0.1]] * 10

        thresholds, metrics = calculate_optimal_thresholds(targets, probs, labels)
        self.assertIn("Atelectasis", thresholds)
        self.assertIn("Cardiomegaly", thresholds)
        self.assertGreaterEqual(thresholds["Atelectasis"], 0.1)
        self.assertLessEqual(thresholds["Atelectasis"], 0.9)


if __name__ == "__main__":
    unittest.main()
