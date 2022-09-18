from invoke import task


@task
def help(c):
    """Print help message."""
    print("Available tasks:")
    print("  - help: Print help message.")
    print("  - venv: Create a virtual environment.")
    print("  - clean: Clean all unnecessary files.")
    print("  - style: Style project.")
    # print("  - docs: Build documentation.")
    print("  - test: Test project.")
    print("  - dvc: Run dvc commands.")


@task
def style(c):
    c.run("black .")
    c.run("flake8")
    c.run("python3 -m isort .")


@task
def venv(c):
    c.run("python3 -m venv venv")
    cmd = 'source venv/bin/activate && python3 -m pip install --upgrade pip setuptools wheel &&  python3 -m pip install -e ".[dev]" && pre-commit install && pre-commit autoupdate'
    c.run(cmd)


@task
def clean(c):
    style(c)
    c.run('find . -type f -name "*.DS_Store" -ls -delete')
    c.run(r'find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf')
    c.run('find . | grep -E ".pytest_cache" | xargs rm -rf')
    c.run('find . | grep -E ".ipynb_checkpoints" | xargs rm -rf')
    c.run('find . | grep -E ".trash" | xargs rm -rf')
    c.run("rm -f .coverage")
    c.run("yes | rm -rf storage/model/.trash/*")


@task
def test(c):
    c.run('pytest -m "not training" --cov=src --cov-report=term-missing --cov-report=xml')
    c.run("cd tests && great_expectations checkpoint run raw")
    c.run("cd tests && great_expectations checkpoint run preprocess")
    c.run("cd tests && great_expectations checkpoint run preprocess_without_outlines")


@task
def dvc(c):
    c.run("dvc add data/smoke_detection_iot.csv")
    c.run("dvc add data/preprocess.csv")
    c.run("dvc add data/preprocess_without_outlines.csv")
    c.run("dvc push")
