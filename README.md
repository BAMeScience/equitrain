# Equitrain: A Unified Framework for Training and Fine-tuning Machine Learning Interatomic Potentials

Equitrain is an open-source software package designed to simplify the training and fine-tuning of machine learning universal interatomic potentials (MLIPs). Equitrain addresses the challenges posed by the diverse and often complex training codes specific to each MLIP by providing a unified and efficient framework. This allows researchers to focus on model development rather than implementation details.

---

## Key Features

- **Unified Framework**: Train and fine-tune MLIPs using a consistent interface.
- **Flexible Backends**: Parity-tested Torch and JAX backends that share schedulers, EMA, and fine-tuning workflows.
- **Flexible Model Wrappers**: Support for different MLIP architectures  (MACE, SevenNet, and ORB) through model-specific wrappers.
- **Efficient Preprocessing**: Automated preprocessing with options for computing statistics and managing data.
- **GPU/Node Scalability**: Seamless integration with multi-GPU and multi-node environments using `accelerate`.
- **Extensive Resources**: Includes scripts for dataset preparation, initial model setup, and training workflows.

---

## Installation

`equitrain` can be installed in your environment by doing:

```bash
pip install equitrain
```

**Note!** Until the package is fully deployed in PyPI, you can only install it by following the instructions below.


### Development

To install the package for development purposes, first clone the repository:

```bash
git clone https://github.com/BAMeScience/equitrain.git
cd equitrain/
```

Create a virtual environment (either with `conda` or `virtualenv`). Note we are using Python 3.10 to create the environment.

**Using `virtualenv`**

Create and activate the environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Make sure `pip` is up-to-date:

```bash
pip install --upgrade pip
```

We recommend using `uv` for the fast installation of the package:

```bash
pip install uv
uv pip install -e '.[dev,docu]'
```

* The `-e` flag makes sure to install the package in editable mode.
* The `[dev]` optional dependencies install a set of packages used for formatting, typing, and testing.
* The `[docu]` optional dependencies install the packages for launching the documentation page.
* For specific model support, you can install additional dependencies:
  * `[orb]` - Install ORB models and dependencies for universal interatomic potentials

**Using `conda`**

Create the environment with the settings file `environment.yml`:

```bash
conda env create -f environment.yml
```

And activate it:

```bash
conda activate equitrain
```

This will automatically install the dependencies. If you want the optional dependencies installed:

```bash
pip install -e '[dev,docu]'
```

Alternatively, you can create a `conda` environment with Python 3.10 and follow all the steps in the installation explained above when using `virtualenv`:

```bash
conda create -n equitrain python=3.10 setuptools pip
conda activate equitrain
```

---

## Quickstart Guide

### 1. Preprocessing Data

Preprocess data files to compute necessary statistics and prepare for training:

#### Command Line:

```bash
equitrain-preprocess \
    --train-file="data-train.xyz" \
    --valid-file="data-valid.xyz" \
    --compute-statistics \
    --atomic-energies="average" \
    --output-dir="data" \
    --r-max 4.5
```

The preprocessing command accepts `.xyz`, `.lmdb`/`.aselmdb`, and `.h5` inputs; LMDB datasets are automatically converted to the native HDF5 format before statistics are computed.

<!-- TODO: change this following a notebook style -->
#### Python Script:

```python
from equitrain import get_args_parser_preprocess, preprocess

def test_preprocess():
    args = get_args_parser_preprocess().parse_args()
    args.train_file         = 'data.xyz'
    args.valid_file         = 'data.xyz'
    args.output_dir         = 'test_preprocess/'
    args.compute_statistics = True
    # Compute atomic energies
    args.atomic_energies    = "average"
    # Cutoff radius for computing graphs
    args.r_max = 4.5

    preprocess(args)

if __name__ == "__main__":
    test_preprocess()
```

---

### 2. Training a Model

Train a model using the prepared dataset and specify the MLIP wrapper:

#### Command Line:

```bash
# Training with MACE
equitrain -v \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir result_mace \
    --model mace.model \
    --model-wrapper 'mace' \
    --epochs 10 \
    --tqdm

# Training with ORB
equitrain -v \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir result_orb \
    --model-wrapper 'orb' \
    --epochs 10 \
    --tqdm
```

<!-- TODO: change this following a notebook style -->
#### Python Script:

```python
from equitrain import get_args_parser_train, train
from equitrain.backends.torch_wrappers import MaceWrapper, OrbWrapper

# Training with MACE
def test_train_mace():
    args = get_args_parser_train().parse_args()
    args.train_file  = 'data/train.h5'
    args.valid_file  = 'data/valid.h5'
    args.output_dir  = 'test_train_mace'
    args.epochs      = 10
    args.batch_size  = 64
    args.lr          = 0.01
    args.verbose     = 1
    args.tqdm        = True
    args.model       = MaceWrapper(args, "mace.model")

    train(args)

if __name__ == "__main__":
    test_train_mace()
    # test_train_orb()
```

#### Running the JAX backend

The training CLI automatically selects the Torch backend. To run the JAX backend instead, point `--backend` to `jax` and provide a JAX bundle exported via `mace_torch2jax` or the new fine-tuning utilities:

```bash
equitrain -v \
    --backend jax \
    --model path/to/jax_bundle \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir result-jax \
    --epochs 5
```

---

### 3. Making Predictions

Use a trained model to make predictions on new data:

<!-- TODO: change this following a notebook style -->
#### Python Script:

```python
from equitrain import get_args_parser_predict, predict
from equitrain.backends.torch_wrappers import MaceWrapper

def test_mace_predict():
    args = get_args_parser_predict().parse_args()
    args.predict_file = 'data/valid.h5'
    args.batch_size   = 64
    args.model        = MaceWrapper(args, "mace.model")

    energy_pred, forces_pred, stress_pred = predict(args)

    print(energy_pred)
    print(forces_pred)
    print(stress_pred)

if __name__ == "__main__":
    test_mace_predict()
```

---

### JAX Backend Multi-Device Notes

- When the JAX backend detects more than one local accelerator, it automatically switches to a multi-device (`pmap`) execution. In that mode the training and evaluation batch size must be divisible by `jax.local_device_count()` so that each device processes an identical number of graphs.
- On single-device machines no extra configuration is required; the backend falls back to the same single-device behaviour that existing scripts expect.

---

### Fine-Tuning with Delta Wrappers

- Additive delta wrappers are available for both backends via `equitrain.finetune`. The Torch helper (`DeltaFineTuneWrapper`) freezes the base model and exposes only the residual parameters for optimisation. The JAX helper (`wrap_with_deltas` / `ensure_delta_params`) provides the same behaviour for Flax modules.
- These utilities power the fine-tuning tests and are ready to be imported in user scripts, enabling LoRA-style workflows without modifying the core training loops.

---

## Advanced Features

### Multi-GPU and Multi-Node Training

Equitrain supports multi-GPU and multi-node training using `accelerate`. Example scripts are available in the `resources/training` directory.

### Dataset Preparation

Equitrain provides scripts for downloading and preparing popular datasets such as Alexandria and MPTraj. These scripts can be found in the `resources/data` directory.

### Pretrained Models

Initial model examples and configurations can be accessed in the `resources/models` directory.
