# Equitrain

Equitrain is a Python toolkit for preprocessing atomistic datasets, training
machine-learning interatomic potentials (MLIPs), fine-tuning existing
checkpoints, and running evaluation or prediction through one CLI/API.

It is designed to keep model-specific code in thin wrappers while sharing the
training, checkpointing, evaluation, prediction, and preprocessing workflow
across supported model families.

The goal is to hide backend-specific training boilerplate behind a consistent
interface so researchers can focus on model and dataset choices rather than
rewriting orchestration code for each MLIP.

## Main Capabilities

- Torch and JAX training entry points with shared argument conventions,
  schedulers, EMA support, and fine-tuning workflows.
- Model wrappers for MACE, SevenNet, ORB, ANI, and M3GNet.
- HDF5 preprocessing for large atomistic datasets.
- Torch reaction-relative losses for barrier and reaction energies.
- Fine-tuning adapters for Delta/L<sup>2</sup>-SP, Freeze, and LoRA workflows.
- ASE calculator helpers for batched prediction and relaxation.
- Multi-GPU and multi-node training templates through Accelerate/JAX
  distributed launch paths.

## Where To Start

- Install Equitrain and optional model extras: [Installation](installation.md)
- Run a first preprocessing/training/evaluation/prediction workflow:
  [Quickstart](quickstart.md)
- Understand input formats, HDF5 layout, metadata, and target keys:
  [Data and Preprocessing](data.md)
- Configure losses, optimizers, checkpointing, freezing, and batching:
  [Training Options](training-options.md)
- Use Equitrain from Python instead of the CLI: [Python API](python-api.md)
- Browse generated public API entries: [API Reference](api-reference.md)
- Prepare JAX model bundles: [JAX Bundles](jax-bundles.md)
- Fine-tune checkpoints with adapters: [Fine-Tuning](fine-tuning.md)
- Reproduce the phonon fine-tuning manuscript workflow:
  [Phonon Fine-Tuning Paper](phonon-finetuning-paper.md)

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
