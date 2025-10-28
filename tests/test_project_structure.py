"""Tests for the ML trading project structure."""

import unittest
import os
import sys
from pathlib import Path


class TestProjectStructure(unittest.TestCase):
    """Test cases for the project structure."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.src_dir = self.project_root / "src" / "ml_trading"

    def test_project_root_exists(self):
        """Test that project root directory exists."""
        self.assertTrue(self.project_root.exists())
        self.assertTrue(self.project_root.is_dir())

    def test_src_directory_exists(self):
        """Test that src directory exists."""
        self.assertTrue(self.src_dir.exists())
        self.assertTrue(self.src_dir.is_dir())

    def test_required_modules_exist(self):
        """Test that required modules exist."""
        required_modules = [
            "config",
            "data",
            "models",
            "pipeline",
            "strategies",
            "utils",
        ]

        for module in required_modules:
            module_path = self.src_dir / module
            self.assertTrue(module_path.exists(), f"Module {module} does not exist")
            self.assertTrue(module_path.is_dir(), f"Module {module} is not a directory")

    def test_required_files_exist(self):
        """Test that required files exist."""
        required_files = [
            "src/ml_trading/__init__.py",
            "src/ml_trading/main.py",
            "src/ml_trading/config/settings.py",
            "src/ml_trading/data/data_loader.py",
            "src/ml_trading/data/feature_engineering.py",
            "src/ml_trading/models/lightgbm_model.py",
            "src/ml_trading/pipeline/multi_tf_pipeline.py",
            "src/ml_trading/pipeline/risk_management.py",
            "src/ml_trading/strategies/ml_strategy.py",
        ]

        for file_path in required_files:
            full_path = self.project_root / file_path
            self.assertTrue(full_path.exists(), f"File {file_path} does not exist")
            self.assertTrue(full_path.is_file(), f"File {file_path} is not a file")

    def test_init_files_exist(self):
        """Test that __init__.py files exist."""
        init_files = [
            "src/ml_trading/__init__.py",
            "src/ml_trading/config/__init__.py",
            "src/ml_trading/data/__init__.py",
            "src/ml_trading/models/__init__.py",
            "src/ml_trading/pipeline/__init__.py",
            "src/ml_trading/strategies/__init__.py",
            "src/ml_trading/utils/__init__.py",
        ]

        for init_file in init_files:
            full_path = self.project_root / init_file
            self.assertTrue(full_path.exists(), f"Init file {init_file} does not exist")
            self.assertTrue(full_path.is_file(), f"Init file {init_file} is not a file")


if __name__ == "__main__":
    unittest.main()
