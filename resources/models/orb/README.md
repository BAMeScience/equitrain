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
from equitrain.model_wrappers import OrbWrapper

# Parse arguments
args = get_args_parser_train().parse_args()

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
from equitrain.model_wrappers import OrbWrapper

# Parse arguments
args = get_args_parser_predict().parse_args()

# Set prediction parameters
args.predict_file = 'data/valid.h5'
args.batch_size = 32

# Create the ORB wrapper
args.model = OrbWrapper(args, model_variant='direct')

# Make predictions
energy_pred, forces_pred, stress_pred = predict(args)
```

### Using a Configuration File

You can also use a YAML configuration file to train an ORB model:

```yaml
# ORB model configuration for Equitrain

# Data paths
train_file: data/train.h5
valid_file: data/valid.h5
output_dir: orb_training

# Model configuration
model_wrapper: orb
model: null  # Will use pretrained ORB model from zoo

# ORB-specific settings
model_variant: direct  # 'direct' or 'conservative'
enable_zbl: false      # Enable ZBL repulsion for high-Z elements (Z > 56)

# Training parameters
epochs: 100
batch_size: 32
lr: 0.001

# Loss weights (ORB defaults)
energy_weight: 0.01
forces_weight: 1.0
stress_weight: 0.1

# Mixed precision (recommended for ORB)
precision: 16
```

Then run:

```bash
equitrain resources/models/orb/orb_config.yaml
```

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

### Graph Compilation
The ORB wrapper automatically initializes graph compilation cache to avoid first-step delays:

```python
# This is done automatically in the wrapper
model = OrbWrapper(args, model_variant='direct')
# Compilation cache is initialized during __init__
```

### Mixed Precision
ORB v3 models were trained in FP16 and work well with automatic mixed precision:

```python
# Enable in your training configuration
precision: 16

# Or programmatically
torch.cuda.amp.autocast(enabled=True)
```

### High-Z Elements
For systems containing elements with Z > 56, enable ZBL repulsion:

```python
model = OrbWrapper(args, enable_zbl=True)
```

## Testing

Run the test suite to verify the ORB integration:

```bash
python tests/test_train_orb.py
```

This test includes:
- Training on a 50-step Aluminum MD slice
- Verification that force MAE < 0.1 eV/Å
- Testing both direct and conservative variants

## References

- [ORB Models GitHub](https://github.com/orbital-materials/orb-models)
- [Orbital Materials](https://orbitalmaterials.com/)
- [ORB Paper](https://arxiv.org/abs/2405.00223) (when available)
