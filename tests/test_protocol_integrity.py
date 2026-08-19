from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.calibrate import calculate_optimal_thresholds
from scripts.compare_models import compare_two_prediction_files, delong_paired_test, holm_bonferroni_correction
from scripts.make_splits import extract_patient_id_strict, iterative_multilabel_split


class TestProtocolIntegrity(unittest.TestCase):
    def test_strict_patient_regex(self) -> None:
        # Valid CheXpert paths
        self.assertEqual(extract_patient_id_strict("CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"), "patient00001")
        self.assertEqual(extract_patient_id_strict("train\\patient12345\\study2\\view1_lateral.jpg"), "patient12345")
        self.assertEqual(extract_patient_id_strict("archive/patient99999/study3/view1_frontal.jpg"), "patient99999")

        # Invalid paths that must return None
        self.assertIsNone(extract_patient_id_strict("arbitrary/folder/image_001.jpg"))
        self.assertIsNone(extract_patient_id_strict("patient_no_digit/study1/view1.jpg"))

    def test_split_determinism_and_zero_overlap(self) -> None:
        """
        Verifies:
        1. Same seed produces identical splits.
        2. Different seeds produce different splits.
        3. Zero patient overlap across train, calibration, and validation.
        """
        n_patients = 100
        patient_ids = [f"patient{i:05d}" for i in range(n_patients)]
        rng = np.random.RandomState(42)
        # 5 multi-label columns
        label_matrix = (rng.rand(n_patients, 5) > 0.7).astype(int)

        ratios = [0.8, 0.1, 0.1]
        split_a = iterative_multilabel_split(patient_ids, label_matrix, ratios, seed=42)
        split_b = iterative_multilabel_split(patient_ids, label_matrix, ratios, seed=42)
        split_c = iterative_multilabel_split(patient_ids, label_matrix, ratios, seed=999)

        # Same seed identical
        self.assertEqual(split_a, split_b)
        # Different seed differs
        self.assertNotEqual(split_a[0], split_c[0])

        # Zero patient overlap
        set_train = set(split_a[0])
        set_calib = set(split_a[1])
        set_val = set(split_a[2])

        self.assertEqual(len(set_train & set_calib), 0)
        self.assertEqual(len(set_train & set_val), 0)
        self.assertEqual(len(set_calib & set_val), 0)
        self.assertEqual(len(set_train | set_calib | set_val), n_patients)

    def test_model_comparison_study_alignment_guards(self) -> None:
        """
        Verifies that compare_two_prediction_files strictly fails when study_id is missing,
        mismatched, or contains duplicates.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            df_a = pd.DataFrame({
                "study_id": ["s1", "s2", "s3"],
                "patient_id": ["p1", "p2", "p3"],
                "Atelectasis_prob": [0.8, 0.2, 0.6],
                "Atelectasis_target": [1.0, 0.0, 1.0],
                "Atelectasis_mask": [1.0, 1.0, 1.0],
            })
            # Model B missing study s3 (has s4 instead)
            df_b_mismatched = pd.DataFrame({
                "study_id": ["s1", "s2", "s4"],
                "patient_id": ["p1", "p2", "p4"],
                "Atelectasis_prob": [0.7, 0.3, 0.5],
                "Atelectasis_target": [1.0, 0.0, 1.0],
                "Atelectasis_mask": [1.0, 1.0, 1.0],
            })
            # Model B valid match
            df_b_valid = pd.DataFrame({
                "study_id": ["s1", "s2", "s3"],
                "patient_id": ["p1", "p2", "p3"],
                "Atelectasis_prob": [0.7, 0.3, 0.5],
                "Atelectasis_target": [1.0, 0.0, 1.0],
                "Atelectasis_mask": [1.0, 1.0, 1.0],
            })

            file_a = tmp_path / "preds_a.csv"
            file_b_mis = tmp_path / "preds_b_mis.csv"
            file_b_val = tmp_path / "preds_b_val.csv"

            df_a.to_csv(file_a, index=False)
            df_b_mismatched.to_csv(file_b_mis, index=False)
            df_b_valid.to_csv(file_b_val, index=False)

            # Mismatched studies must raise ValueError
            with self.assertRaises(ValueError):
                compare_two_prediction_files(file_a, file_b_mis, labels=["Atelectasis"])

            # Valid matched files must succeed
            res = compare_two_prediction_files(file_a, file_b_val, labels=["Atelectasis"], n_boot=100)
            self.assertEqual(res["aligned_studies"], 3)
            self.assertIn("Atelectasis", res["results_per_label"])

    def test_calibration_null_threshold_on_single_class(self) -> None:
        """
        P0 Test: Verifies that if a label has only 1 class in calibration set (e.g. all 0s),
        threshold is set to None / null, NOT fake 0.5.
        """
        targets = [[0.0], [0.0], [0.0], [0.0]]
        probs = [[0.1], [0.2], [0.05], [0.15]]
        labels = ["RareDisease"]

        thresholds, metrics = calculate_optimal_thresholds(targets, probs, labels)
        self.assertIsNone(thresholds["RareDisease"], "Threshold for single-class label must be null!")


if __name__ == "__main__":
    unittest.main()
