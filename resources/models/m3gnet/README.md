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
pip install 'matgl>=4.0.0'
```

## Usage

### Training a M3GNet Model

You can train a M3GNet model using the Equitrain framework:

```python
from equitrain import get_args_parser_train, train
from equitrain.backends.torch_wrappers import M3GNetWrapper

# Parse arguments
args = get_args_parser_train().parse_args([])

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
from equitrain.backends.torch_wrappers import M3GNetWrapper

# Parse arguments
args = get_args_parser_predict().parse_args([])

# Set prediction parameters
args.predict_file = 'data/valid.h5'
args.batch_size = 32

# Create the M3GNet wrapper
args.model = M3GNetWrapper(args)

# Make predictions
energy_pred, forces_pred, stress_pred = predict(args)
```

### Configuration Sketch

`m3gnet-config.yaml` records the relevant settings for notebooks, custom
launchers, or external configuration systems. The current `equitrain` console
script does not read YAML files directly, so pass the equivalent values as CLI
flags or load them into an `argparse.Namespace` in Python.

CLI shape:

```bash
equitrain -v \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir m3gnet_training \
    --model resources/models/m3gnet/m3gnet-initial-model.pt \
    --model-wrapper m3gnet \
    --energy-weight 1.0 \
    --forces-weight 10.0 \
    --stress-weight 0.1 \
    --loss-type mse
```

## References

- [MatGL Documentation](https://matgl.ai/)
- [M3GNet Paper](https://nature.com/articles/s43588-022-00349-3)
