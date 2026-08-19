from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestSecretLeakageAndNotebookIntegrity(unittest.TestCase):
    def test_no_kaggle_secrets_or_tokens_in_codebase(self) -> None:
        """Requirement 4: Verify zero Kaggle tokens, KGAT_ tokens, or leaked secrets in repo."""
        forbidden_patterns = [
            re.compile(r"KAGGLE_TOKEN\s*="),
            re.compile(r"KGAT_[a-zA-Z0-9_-]{10,}"),
            re.compile(r'"key":\s*"[0-9a-f]{32}"'),
            re.compile(r"google\.colab"),
        ]

        ignored_extensions = {".png", ".pt", ".jpg", ".jpeg", ".zip", ".pyc", ".pdf"}
        ignored_files = {"test_secret_leakage.py", "ci.yml"}

        for filepath in PROJECT_ROOT.rglob("*"):
            if not filepath.is_file():
                continue
            if filepath.suffix in ignored_extensions:
                continue
            if any(part.startswith(".") for part in filepath.parts if part not in [".github"]):
                continue
            if filepath.name in ignored_files:
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for pattern in forbidden_patterns:
                match = pattern.search(content)
                self.assertIsNone(
                    match,
                    f"SECURITY VIOLATION: Forbidden token pattern '{pattern.pattern}' found in {filepath.relative_to(PROJECT_ROOT)}!",
                )

    def test_train_on_kaggle_notebook_structure_and_cleanliness(self) -> None:
        """Verifies train_on_kaggle.ipynb format, absence of Colab code, and clean execution counts."""
        nb_path = PROJECT_ROOT / "train_on_kaggle.ipynb"
        self.assertTrue(nb_path.is_file(), "train_on_kaggle.ipynb must exist!")

        nb_content = nb_path.read_text(encoding="utf-8")
        nb_json = json.loads(nb_content)

        self.assertIn("cells", nb_json)
        self.assertGreaterEqual(len(nb_json["cells"]), 10)

        # Check that outputs are empty and execution_count is None
        for idx, cell in enumerate(nb_json["cells"]):
            if cell.get("cell_type") == "code":
                self.assertIsNone(
                    cell.get("execution_count"),
                    f"Cell {idx} has non-null execution_count in train_on_kaggle.ipynb",
                )
                self.assertEqual(
                    cell.get("outputs"),
                    [],
                    f"Cell {idx} has non-empty outputs in train_on_kaggle.ipynb",
                )
                source_code = "".join(cell.get("source", []))
                self.assertNotIn("google.colab", source_code)
                self.assertNotIn("drive.mount", source_code)


if __name__ == "__main__":
    unittest.main()
