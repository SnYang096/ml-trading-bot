"""Setup script for the ML Trading Project."""

from setuptools import setup, find_packages

import os

# Read README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "ML Trading Project"

# Read requirements
requirements = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]
else:
    # Minimal requirements if file doesn't exist
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "lightgbm>=3.3.0",
    ]

setup(
    name="ml-trading-project",
    version="0.0.2",
    author="Your Name",
    author_email="your.email@example.com",
    description="Machine learning algorithmic trading system with multi-timeframe analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-trading-project",
    packages=find_packages(where="src") + find_packages(where=".", include=["scripts*"]),
    package_dir={
        "": "src",
        "scripts": "scripts",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Unified CLI (recommended)
            # Note: package_dir sets "" to "src", so we use cli.main, not src.cli.main
            "mlbot=cli.main:main",
            # Legacy scripts for backward compatibility
            "train-strategy=scripts.train_strategy_pipeline:main",
        ],
    },
)
