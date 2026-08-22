# ORB Wrapper for Equitrain

This directory contains resources for using ORB (Orbital Materials) universal interatomic potential models with Equitrain.

## Overview

ORB is a universal, PyTorch-native interatomic potential family released by Orbital Materials. Version 3 models (April 2025) compile with PyTorch ≥2.6 and run 2-3× faster than v2 while cutting Matbench-Discovery error by ≈31%. The ORB wrapper in Equitrain allows you to use ORB models within the Equitrain framework for training, evaluation, and prediction.

### Key Features

- **Universal Coverage**: Supports >80 elements (H to Po)
- **High Performance**: 2-3× faster than previous versions with PyTorch compilation
- **Two Variants**:
  - **Direct**: Forwards also output per-atom forces + stress
  - **Conservative**: Only energy; forces/stress via torch.autograd.grad
- **ZBL Repulsion**: Optional ZBL repulsion term for high-Z elements (Z > 56)
- **Mixed Precision**: Supports FP16 training with automatic mixed precision

## Installation

To use the ORB wrapper, you need to install the ORB models library:

```bash
pip install equitrain[orb]
```

Or manually:

```bash
pip install "orb-models>=3.0"
pip install "cuml-cu11"  # Optional, speeds neighbor graph build on GPU (Linux only)
```

## Usage

### Training an ORB Model

You can train an ORB model using the Equitrain framework:

```python
from equitrain import get_args_parser_train, train
from equitrain.backends.torch_wrappers import OrbWrapper

# Parse arguments
args = get_args_parser_train().parse_args([])

# Set training parameters
args.train_file = 'data/train.h5'
args.valid_file = 'data/valid.h5'
args.output_dir = 'train_orb'
args.epochs = 100
args.batch_size = 32
args.lr = 0.001
args.verbose = 1
args.tqdm = True

# Set loss weights (ORB defaults: 0.01 × energy + 1.0 × forces + 0.1 × stress)
args.energy_weight = 0.01
args.forces_weight = 1.0
args.stress_weight = 0.1

# Create the ORB wrapper
args.model = OrbWrapper(args, model_variant='direct', enable_zbl=False)

# Train the model
train(args)
```

### Making Predictions with an ORB Model

You can use a trained ORB model to make predictions:

```python
from equitrain import get_args_parser_predict, predict
from equitrain.backends.torch_wrappers import OrbWrapper

# Parse arguments
args = get_args_parser_predict().parse_args([])

# Set prediction parameters
args.predict_file = 'data/valid.h5'
args.batch_size = 32

# Create the ORB wrapper
args.model = OrbWrapper(args, model_variant='direct')

# Make predictions
energy_pred, forces_pred, stress_pred = predict(args)
```

### Configuration Sketch

`orb_config.yaml` records the relevant settings for notebooks, custom launchers,
or external configuration systems. The current `equitrain` console script does
not read YAML files directly, so pass the equivalent values as CLI flags or load
them into an `argparse.Namespace` in Python.

CLI shape:

```bash
equitrain -v \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir orb_training \
    --model path/to/orb.model \
    --model-wrapper orb \
    --energy-weight 0.01 \
    --forces-weight 1.0 \
    --stress-weight 0.1 \
    --loss-type mse
```

ORB-specific constructor options such as `model_variant` and `enable_zbl` are
Python-level wrapper options. For CLI training, save or load a model artifact
that already has the desired ORB behavior.

## Model Variants

### Direct Variant
- Outputs energy, forces, and stress directly from the forward pass
- Faster inference
- Recommended for most use cases

### Conservative Variant
- Only outputs energy from the forward pass
- Forces and stress computed via `torch.autograd.grad`
- More memory efficient for training
- Useful when only energy is needed

## Performance Optimization

### Model Loading
When `model=None`, `OrbWrapper` tries the available ORB pretrained factories for
the requested variant. For CLI workflows, pass a concrete model artifact via
`--model` so `equitrain` can load it before wrapping.

### Mixed Precision
The ORB wrapper uses `torch.amp.autocast('cuda')` during the wrapped forward pass
when CUDA is available. Equitrain does not currently expose a top-level
`--precision` flag for ORB training.

### High-Z Elements
For systems containing elements with Z > 56, enable ZBL repulsion:

```python
model = OrbWrapper(args, enable_zbl=True)
```

## Testing

Run the test suite to verify the ORB integration:

```bash
pytest tests/test_train_orb.py
```

This test includes:
- Training on a 50-step Aluminum MD slice
- Verification that force MAE < 0.1 eV/Å
- Testing both direct and conservative variants

## References

- [ORB Models GitHub](https://github.com/orbital-materials/orb-models)
- [Orbital Materials](https://orbitalmaterials.com/)
- [ORB Paper](https://arxiv.org/abs/2405.00223) (when available)
