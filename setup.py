# setup.py
from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# Define build requirements
build_packages = ["pre-commit==2.20.0", "invoke==1.7.1", "nox==2022.8.7"]

# Define docs requirements
docs_packages = ["mkdocs==1.3.1", "mkdocstrings==0.19.0"]

# Define stylings requirements
style_packages = ["black==22.6.0", "flake8==5.0.4", "isort==5.10.1"]

# Define test requirements
test_packages = [
    "pytest==7.1.2",
    "pytest-cov==3.0.0",
    "pytest-mock==3.8.2",
    "great-expectations==0.15.20",
    "pretty_errors==1.2.25",
    "mypy==0.971",
    "types-requests==2.28.10",
]

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
        "dev": docs_packages + style_packages + test_packages + build_packages,
        "docs": docs_packages,
        "test": test_packages,
        "style": style_packages,
    },
)
