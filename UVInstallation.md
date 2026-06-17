# Installation (uv)

Volume Segmantics now uses [uv](https://docs.astral.sh/uv/) instead of Poetry for dependency management and installation. uv is faster and replaces both Poetry and pip for this project.

## 1. Install uv (Mac & Linux)

Install uv at the system/user level **not** inside a conda environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
if you don't have `curl`, you can use `wget`

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

## 2. Clone the repository

```bash
git clone https://github.com/rosalindfranklininstitute/volume-segmantics
cd volume-segmantics
```

## 3. Install the package

Standard install, including dev dependencies:

```bash
uv sync --dev
```

This creates a `.venv` folder in the project directory and installs everything needed to develop and test the package, generating a `uv.lock` file.

## 4. Activate the environment

```bash
source .venv/bin/activate
```
## Configuration and command line use
After installation, two new commands will be available from your terminal whilst your environment is activated, `model-train-2d` and `model-predict-2d`.

## Notes
- To add a new runtime dependency: `uv add <package-name>`
- To add a new dev dependency: `uv add --dev <package-name>`