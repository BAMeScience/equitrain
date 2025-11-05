# M3GNet Wrapper for Equitrain

This directory contains resources for using the M3GNet (Materials 3-body Graph Network) model from the MatGL library with Equitrain.

## Overview

M3GNet is a graph neural network model for materials science that incorporates 3-body interactions. It is designed to predict energy, forces, and stress for atomic systems. The M3GNet wrapper in Equitrain allows you to use M3GNet models within the Equitrain framework for training, evaluation, and prediction.

## Installation

To use the M3GNet wrapper, you need to install the MatGL library and its dependencies:

```bash
pip install equitrain[m3gnet]
```

Or manually:

```bash
pip install matgl>=1.0.0 dgl>=1.0.0
```

## Usage

### Training a M3GNet Model

You can train a M3GNet model using the Equitrain framework:

```python
from equitrain import get_args_parser_train, train
from equitrain.utility_test import M3GNetWrapper

# Parse arguments
args = get_args_parser_train().parse_args()

# Set training parameters
args.train_file = 'data/train.h5'
args.valid_file = 'data/valid.h5'
args.output_dir = 'train_m3gnet'
args.epochs = 100
args.batch_size = 32
args.lr = 0.001
args.verbose = 1
args.tqdm = True

# Set loss weights
args.energy_weight = 1.0
args.forces_weight = 10.0
args.stress_weight = 0.1

# Create the M3GNet wrapper
args.model = M3GNetWrapper(args)

# Train the model
train(args)
```

### Making Predictions with a M3GNet Model

You can use a trained M3GNet model to make predictions:

```python
from equitrain import get_args_parser_predict, predict
from equitrain.utility_test import M3GNetWrapper

# Parse arguments
args = get_args_parser_predict().parse_args()

# Set prediction parameters
args.predict_file = 'data/valid.h5'
args.batch_size = 32

# Create the M3GNet wrapper
args.model = M3GNetWrapper(args)

# Make predictions
energy_pred, forces_pred, stress_pred = predict(args)
```

### Using a Configuration File

You can also use a YAML configuration file to train a M3GNet model:

```yaml
# M3GNet model configuration for Equitrain

# Data paths
train_file: data/train.h5
valid_file: data/valid.h5
output_dir: m3gnet_training

# Model configuration
model_wrapper: m3gnet
model: resources/models/m3gnet/m3gnet-initial-model.pt

# Training parameters
epochs: 100
batch_size: 32
lr: 0.001
verbose: 1
tqdm: true

# Loss weights
energy_weight: 1.0
forces_weight: 10.0
stress_weight: 0.1

# Loss function
loss_type: mse
loss_energy_per_atom: true
```
Then run:
```bash
equitrain resources/models/m3gnet/m3gnet_config.yaml
```

## References

- [MatGL Documentation](https://matgl.ai/)
- [M3GNet Paper](https://nature.com/articles/s43588-022-00349-3)
