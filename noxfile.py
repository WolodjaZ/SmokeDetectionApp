import nox

SUPPORTED_PY_VERSIONS = ["3.8", "3.9", "3.10"]
nox.options.sessions = ["lint", "test", "coverage", "mypy"]


def _deps(session: nox.Session) -> None:
    session.install("--upgrade", "setuptools", "pip", "wheel")
    session.install("pre-commit")


def _install_dev_packages(session):
    session.install("-e", ".[dev]")


def _install_test_dependencies(session):
    session.install("-e", ".[test]")


def _install_doc_dependencies(session):
    session.install("-e", ".[docs]")


def _install_style_dependencies(session):
    session.install("-e", ".[style]")


@nox.session(python=SUPPORTED_PY_VERSIONS)
def test(session: nox.Session) -> None:
    """Pytesting."""
    _deps(session)
    _install_dev_packages(session)
    _install_test_dependencies(session)

    session.run("pytest", "-m", "not training")
    session.notify("great_expectations")


@nox.session
def great_expectations(session: nox.Session) -> None:
    _deps(session)
    _install_test_dependencies(session)
    with session.chdir("tests"):
        session.run("great_expectations", "checkpoint", "run", "raw")
        session.run("great_expectations", "checkpoint", "run", "preprocess")
        session.run("great_expectations", "checkpoint", "run", "preprocess_without_outlines")


@nox.session(python=SUPPORTED_PY_VERSIONS)
def coverage(session: nox.Session) -> None:
    """Coverage analysis."""
    _deps(session)
    _install_test_dependencies(session)
    session.run("coverage", "erase")
    session.run("coverage", "run", "--append", "-m", "pytest", "-m", "not training")
    session.run("coverage", "report", "--fail-under=20")  # 100
    session.run("coverage", "erase")
    session.notify("great_expectations")


@nox.session(python=SUPPORTED_PY_VERSIONS)
def lint(session: nox.Session) -> None:
    """Style check."""
    _deps(session)
    _install_style_dependencies(session)
    session.run("isort", ".", "--check")
    session.run("black", "--extend-exclude", "venv|.nox", ".")
    session.run("flake8")


@nox.session(python=SUPPORTED_PY_VERSIONS)
def mypy(session: nox.Session) -> None:
    """Type check."""
    _deps(session)
    _install_test_dependencies(session)
    """Run mypy."""
    session.run(
        "mypy",
        "tasks.py",
        "noxfile.py",
        "src",
        "tests",
        "app",
    )
