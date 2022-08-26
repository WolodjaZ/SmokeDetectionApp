# setup.py
from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# Define docs requirements
docs_packages = ["mkdocs==1.3.1", "mkdocstrings==0.19.0"]

# Define stylings requirements
style_packages = ["black==22.6.0", "flake8==5.0.4", "isort==5.10.1"]

# Define test requirements
test_packages = ["pytest==7.1.2", "pytest-cov==3.0.0", "great-expectations==0.15.20"]

# Define our package
setup(
    name="SmokeDetector",
    version=0.1,
    description="Classify smoke detection.",
    author="Vladimir Zaigrajew",
    author_email="vladimirzaigrajew@gmail.com",
    python_requires=">=3.8",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "dev": docs_packages + style_packages + test_packages,
        "docs": docs_packages,
        "test": test_packages,
    },
)
