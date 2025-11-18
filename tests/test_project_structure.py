"""Tests validating the reorganised project structure."""

from pathlib import Path
import unittest


class TestProjectStructure(unittest.TestCase):
    """Ensure key directories and modules exist after package reorg."""

    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.time_series_dir = self.project_root / "src" / "time_series_model"
        self.data_tools_dir = self.project_root / "src" / "data_tools"
        self.cross_sectional_dir = self.project_root / "src" / "cross_sectional"

    def test_core_directories_exist(self):
        for path in (
            self.time_series_dir,
            self.data_tools_dir,
            self.cross_sectional_dir,
        ):
            with self.subTest(path=path):
                self.assertTrue(path.exists(), f"{path} is missing")
                self.assertTrue(path.is_dir(), f"{path} is not a directory")

    def test_time_series_modules_exist(self):
        required = [
            "backtesting",
            "config",
            "feature_management",
            "models",
            "pipeline",
            "strategies",
            "utils",
        ]
        for module in required:
            module_path = self.time_series_dir / module
            with self.subTest(module=module):
                self.assertTrue(module_path.exists(), f"Module {module} does not exist")
                self.assertTrue(
                    module_path.is_dir(), f"Module {module} is not a directory"
                )

    def test_key_files_exist(self):
        required_files = [
            "src/time_series_model/__init__.py",
            "src/time_series_model/config/settings.py",
            "src/time_series_model/models/lightgbm_model.py",
            "src/time_series_model/pipeline/multi_tf_pipeline.py",
            "src/time_series_model/pipeline/risk_management.py",
            "src/time_series_model/strategies/ml_strategy.py",
            "src/data_tools/data_loader.py",
            "src/data_tools/baseline_feature_engineering.py",
            "src/cross_sectional/panel.py",
        ]
        for rel_path in required_files:
            full_path = self.project_root / rel_path
            with self.subTest(path=rel_path):
                self.assertTrue(full_path.exists(), f"{rel_path} missing")
                self.assertTrue(full_path.is_file(), f"{rel_path} is not a file")

    def test_init_files_exist(self):
        init_files = [
            "src/time_series_model/__init__.py",
            "src/time_series_model/config/__init__.py",
            "src/time_series_model/models/__init__.py",
            "src/time_series_model/pipeline/__init__.py",
            "src/time_series_model/strategies/__init__.py",
            "src/time_series_model/utils/__init__.py",
            "src/data_tools/__init__.py",
            "src/cross_sectional/__init__.py",
        ]
        for rel_path in init_files:
            full_path = self.project_root / rel_path
            with self.subTest(path=rel_path):
                self.assertTrue(full_path.exists(), f"{rel_path} missing")
                self.assertTrue(full_path.is_file(), f"{rel_path} is not a file")


if __name__ == "__main__":
    unittest.main()
