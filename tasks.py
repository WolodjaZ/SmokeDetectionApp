from invoke import task

CURRENT_PYTHON_VERSION = "3.8"


@task
def help(c):
    """Print help message."""
    print("Available tasks:")
    print("  - help: Print help message.")
    print("  - venv: Create a virtual environment.")
    print("  - clean: Clean all unnecessary files.")
    print("  - style: Style project.")
    print("  - test: Test project.")
    print("  - coverage: Coverage analysis.")
    print("  - mypy: Typing analysis.")
    print("  - dvc: Run dvc commands.")
    # print("  - docs: Build documentation.")


@task
def style(c, python_version=CURRENT_PYTHON_VERSION):
    c.run(f"nox -s lint-{python_version}")


@task
def venv(c):
    c.run("python3 -m venv venv")
    cmd = 'source venv/bin/activate && python3 -m pip install --upgrade pip setuptools wheel &&  python3 -m pip install -e ".[dev]" && pre-commit install && pre-commit autoupdate'
    c.run(cmd)


@task
def clean(c, python_version=CURRENT_PYTHON_VERSION):
    c.run('find . -type f -name "*.DS_Store" -ls -delete')
    c.run(r'find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf')
    c.run('find . | grep -E ".pytest_cache" | xargs rm -rf')
    c.run('find . | grep -E ".ipynb_checkpoints" | xargs rm -rf')
    c.run('find . | grep -E ".trash" | xargs rm -rf')
    c.run("rm -f .coverage")
    c.run("yes | rm -rf storage/model/.trash/*")


@task
def test(c, python_version=CURRENT_PYTHON_VERSION):
    c.run(f"nox -s test-{python_version}")


@task
def coverage(c, python_version=CURRENT_PYTHON_VERSION):
    c.run(f"nox -s coverage-{python_version}")


@task
def mypy(c, python_version=CURRENT_PYTHON_VERSION):
    c.run(f"nox -s mypy-{python_version}")


@task
def dvc(c):
    c.run("dvc add data/smoke_detection_iot.csv")
    c.run("dvc add data/preprocess.csv")
    c.run("dvc add data/preprocess_without_outlines.csv")
    c.run("dvc add results/performance.json")
    c.run("dvc push")
