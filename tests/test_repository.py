"""Dependency-free source checks."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SmokeTests(unittest.TestCase):
    def test_entrypoint_syntax(self):
        ast.parse((ROOT / "river_app_final.py").read_text(encoding="utf-8"))

    def test_dataset_paths(self):
        config = (ROOT / "data.yaml").read_text(encoding="utf-8")
        self.assertIn("path: ./dataset", config)
        self.assertIn("train: train/images", config)
        self.assertIn("val: valid/images", config)
        self.assertNotIn("D:\\", config)

    def test_threaded_batch_and_cleanup(self):
        source = (ROOT / "river_app_final.py").read_text(encoding="utf-8")
        for token in ("class BatchWorker(QThread)", "stop_event", "closeEvent",
                      "cv2.imdecode", "cv2.imencode", "draw_idle"):
            self.assertIn(token, source)

    def test_dependencies_and_docs_exist(self):
        self.assertTrue((ROOT / "requirements.txt").is_file())
        self.assertIn("best.pt", (ROOT / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
