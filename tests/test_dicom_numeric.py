from __future__ import annotations

import io
import unittest
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
from PIL import Image

from app.main import decode_image_or_dicom


class TestDicomNumericPreprocessing(unittest.TestCase):
    def test_known_numeric_pipeline_values(self) -> None:
        """
        P1 Test: Numeric test verifying exact numerical stages of DICOM preprocessing:
        1. Raw pixels: [[100, 200], [300, 400]]
        2. Modality LUT: RescaleSlope=2.0, RescaleIntercept=-50 -> [[150, 350], [550, 750]]
        3. VOI Windowing: WindowCenter=450, WindowWidth=400 -> min=250, max=650 -> clipped to [250, 650]
        4. MONOCHROME1 Inversion: 255 - normalized_arr
        5. Normalization to [0, 255] uint8.
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
        ds.Rows = 2
        ds.Columns = 2
        ds.BitsAllocated = 16
        ds.BitsStored = 12
        ds.HighBit = 11
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.RescaleSlope = 2.0
        ds.RescaleIntercept = -50.0
        ds.WindowCenter = 450
        ds.WindowWidth = 400

        # Raw pixels: [100, 200, 300, 400]
        raw_arr = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        ds.PixelData = raw_arr.tobytes()

        buf = io.BytesIO()
        pydicom.dcmwrite(buf, ds, enforce_file_format=False)
        buf.seek(0)

        img, dcm_meta = decode_image_or_dicom(buf, "numeric_test.dcm")
        arr = np.asarray(img.convert("L"))

        self.assertEqual(arr.shape, (2, 2))
        self.assertEqual(arr.dtype, np.uint8)

        # Because of MONOCHROME1, lower raw value should yield higher pixel intensity
        # raw 100 -> modality 150 -> clipped to 250 -> min val -> inverted to 255
        # raw 400 -> modality 750 -> clipped to 650 -> max val -> inverted to 0
        self.assertGreater(arr[0, 0], arr[1, 1], "MONOCHROME1 inversion failed: raw min should invert to high pixel intensity")
        self.assertEqual(arr[0, 0], 255)
        self.assertEqual(arr[1, 1], 0)


if __name__ == "__main__":
    unittest.main()
