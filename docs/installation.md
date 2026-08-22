# Installation

## Package Install

```bash
pip install equitrain
```

Until the package is fully available on PyPI, install from a local clone:

```bash
git clone https://github.com/BAMeScience/equitrain.git
cd equitrain
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install uv
uv pip install -e '.[dev,docu]'
```

The `-e` flag installs the package in editable mode. The `dev` extra installs
formatting, typing, and testing tools. The `docu` extra installs the
documentation build stack.

## Optional Extras

Install only the runtime stacks needed for the model families you use.

| Extra | Purpose |
| --- | --- |
| `torch` | Core Torch backend: PyTorch, PyG, Accelerate, EMA. |
| `jax` | Core JAX backend runtime. |
| `mace` | Torch MACE support. |
| `mace-jax` | JAX MACE support. |
| `ani` | TorchANI support. |
| `orb` | ORB model support. |
| `m3gnet` | MatGL/M3GNet support. |

Examples:

```bash
pip install 'equitrain[torch,mace]'
pip install 'equitrain[jax,mace-jax]'
pip install 'equitrain[torch,ani]'
```

## Development Environment

A conda environment file is provided:

```bash
conda env create -f environment.yml
conda activate equitrain
pip install -e '.[dev,docu]'
```

The `dev` extra installs test and formatting tools. The `docu` extra installs
MkDocs and the documentation theme used by this site.

Alternatively, create a minimal conda environment and install the package
manually:

```bash
conda create -n equitrain python=3.10 setuptools pip
conda activate equitrain
pip install -e '.[dev,docu]'
```

## Building These Docs Locally

```bash
pip install -e '.[docu]'
mkdocs serve
```

For a static build:

```bash
mkdocs build
```
