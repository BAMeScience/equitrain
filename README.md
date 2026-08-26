# Equitrain: Training and Fine-Tuning Machine Learning Interatomic Potentials

Equitrain is a Python toolkit for preprocessing atomistic datasets, training
machine-learning interatomic potentials (MLIPs), fine-tuning existing
checkpoints, and running evaluation or prediction through one CLI/API.

## Features

- Unified Torch and JAX training entry points.
- Model wrappers for MACE, SevenNet, ORB, ANI, and M3GNet.
- Native HDF5 preprocessing for large atomistic datasets.
- Torch reaction-relative losses for barrier and reaction energies.
- Fine-tuning adapters for Delta/L<sup>2</sup>-SP, Freeze, and LoRA workflows.
- ASE calculator helpers for batched prediction and relaxation.

## Supported Models

| Wrapper | Backends | Upstream / Companion Project | Notes |
| --- | --- | --- | --- |
| `mace` | Torch, JAX | [`mace-model`](https://github.com/bamescience/mace-model) | Companion repository for MACE model definitions, conversion, and foundation-model export. |
| `sevennet` | Torch | [`MDIL-SNU/SevenNet`](https://github.com/MDIL-SNU/SevenNet) | Torch SevenNet checkpoints and models. |
| `orb` | Torch | [`orbital-materials/orb-models`](https://github.com/orbital-materials/orb-models) | Torch ORB force-field models. |
| `ani` | Torch, JAX | [`aiqm/torchani`](https://github.com/aiqm/torchani) | Torch uses TorchANI; JAX uses a JAX-native bundle. |
| `m3gnet` | Torch, JAX | [`materialsvirtuallab/matgl`](https://github.com/materialsvirtuallab/matgl) | Torch uses MatGL; JAX uses a JAX-native bundle. |

For MACE, use [`mace-model`](https://github.com/bamescience/mace-model) for
model construction/conversion and `equitrain` for preprocessing, training,
fine-tuning, checkpointing, evaluation, and prediction.

## Documentation

Full documentation is published at
[https://bamescience.github.io/equitrain/](https://bamescience.github.io/equitrain/):

- [Installation](https://bamescience.github.io/equitrain/installation/)
- [Quickstart](https://bamescience.github.io/equitrain/quickstart/)
- [Data and Preprocessing](https://bamescience.github.io/equitrain/data/)
- [CLI](https://bamescience.github.io/equitrain/cli/)
- [Training Options](https://bamescience.github.io/equitrain/training-options/)
- [Python API](https://bamescience.github.io/equitrain/python-api/)
- [API Reference](https://bamescience.github.io/equitrain/api-reference/)
- [Model Wrappers](https://bamescience.github.io/equitrain/model-wrappers/)
- [JAX Bundles](https://bamescience.github.io/equitrain/jax-bundles/)
- [Fine-Tuning](https://bamescience.github.io/equitrain/fine-tuning/)
- [Phonon Fine-Tuning Paper](https://bamescience.github.io/equitrain/phonon-finetuning-paper/)
- [Calculators](https://bamescience.github.io/equitrain/calculators/)
- [Reaction-Relative Losses](https://bamescience.github.io/equitrain/reaction-relative-losses/)
- [Resources](https://bamescience.github.io/equitrain/resources/)

The documentation source is in `docs/`. Build or serve it locally with:

```bash
pip install -e '.[docu]'
mkdocs serve
```

## Installation

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

Install model/runtime extras as needed:

```bash
pip install 'equitrain[torch,mace]'
pip install 'equitrain[jax,mace-jax]'
pip install 'equitrain[torch,ani]'
```

## Minimal Workflow

Preprocess data:

```bash
equitrain-preprocess \
    --train-file data-train.xyz \
    --valid-file data-valid.xyz \
    --compute-statistics \
    --atomic-energies average \
    --output-dir data \
    --r-max 4.5
```

Train a Torch/MACE model:

```bash
equitrain -v \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir runs/mace \
    --model path/to/mace.model \
    --model-wrapper mace \
    --epochs 10 \
    --tqdm
```

Evaluate and predict:

```bash
equitrain-evaluate -v \
    --test-file data/test.h5 \
    --model path/to/mace.model \
    --model-wrapper mace \
    --output-dir evaluation_mace

equitrain-predict \
    --predict-file data/valid.h5 \
    --model path/to/mace.model \
    --model-wrapper mace \
    --output-dir predictions_mace
```

See the [Quickstart](https://bamescience.github.io/equitrain/quickstart/) for the full workflow, including JAX
bundles and fine-tuned checkpoint export.

## Fine-Tuning Note

Equitrain's Delta adapter is a residual-parameter implementation of
L<sup>2</sup>-SP ("Starting Point") regularization from Li, Grandvalet, and
Davoine, 2018,
[*Explicit Inductive Bias for Transfer Learning with Convolutional Networks*](https://proceedings.mlr.press/v80/li18a.html).
It parameterizes fine-tuning as `theta = theta_0 + delta`, so weight decay on
trainable deltas regularizes `||delta||_2^2`.

Delta combined with `freeze_layers` is targeted L<sup>2</sup>-SP
(L<sup>2</sup>-TSP): the L<sup>2</sup>-SP penalty applies only to selected
trainable delta layers while frozen layers remain exactly at their pre-trained
starting values. See [Fine-Tuning](https://bamescience.github.io/equitrain/fine-tuning/).

## Resources

Example data-preparation scripts are in `resources/data`, training scripts are
in `resources/training`, and initial model examples are in `resources/models`.
