"""Script to initialize the ML trading project structure."""

import os
import sys
from pathlib import Path


def create_directory_structure():
    """Create the required directory structure."""
    directories = [
        "src/ml_trading",
        "src/ml_trading/config",
        "src/ml_trading/data",
        "src/ml_trading/models",
        "src/ml_trading/pipeline",
        "src/ml_trading/strategies",
        "src/ml_trading/utils",
        "tests",
        "examples",
        "docs",
        "logs",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def create_init_files():
    """Create __init__.py files for Python packages."""
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
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write('"""Initialization file."""\n')
            print(f"Created init file: {init_file}")


def main():
    """Main function to initialize the project."""
    print("Initializing ML Trading Project structure...")

    try:
        create_directory_structure()
        create_init_files()

        print("\nProject structure initialized successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the main script: python src/ml_trading/main.py")
        print("3. Check the documentation in docs/ for more information")

    except Exception as e:
        print(f"Error initializing project: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
