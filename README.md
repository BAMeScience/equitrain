# Equitrain: A Unified Framework for Training and Fine-tuning Machine Learning Interatomic Potentials

Equitrain is a Python toolkit for preprocessing atomistic datasets, training interatomic potential models, and fine-tuning existing checkpoints through one consistent CLI and API.

Equitrain is an open-source software package designed to simplify the training and fine-tuning of machine learning universal interatomic potentials (MLIPs). Equitrain addresses the challenges posed by the diverse and often complex training codes specific to each MLIP by providing a unified and efficient framework. This allows researchers to focus on model development rather than implementation details.

---

## Key Features

- **Unified Framework**: Train and fine-tune MLIPs using a consistent interface.
- **Flexible Backends**: Parity-tested Torch and JAX backends that share schedulers, EMA, and fine-tuning workflows.
- **Flexible Model Wrappers**: Support for different MLIP architectures  (MACE, SevenNet, ORB, ANI, and M3GNet) through model-specific wrappers.
- **Efficient Preprocessing**: Automated preprocessing with options for computing statistics and managing data.
- **GPU/Node Scalability**: Seamless integration with multi-GPU and multi-node environments using `accelerate`.
- **Extensive Resources**: Includes scripts for dataset preparation, initial model setup, and training workflows.

---

## Supported Models

`equitrain` currently supports the following model families through model
wrappers:

| Wrapper | Backends | Upstream / Companion Project | Notes |
| --- | --- | --- | --- |
| `mace` | Torch, JAX | [`mace-model`](https://github.com/bamescience/mace-model) | Companion repository in this workspace for MACE model definitions, conversion, and foundation-model export. |
| `sevennet` | Torch | [`MDIL-SNU/SevenNet`](https://github.com/MDIL-SNU/SevenNet) | Torch wrapper around SevenNet checkpoints and models. |
| `orb` | Torch | [`orbital-materials/orb-models`](https://github.com/orbital-materials/orb-models) | Torch wrapper around ORB force-field models. |
| `ani` | Torch, JAX | [`aiqm/torchani`](https://github.com/aiqm/torchani) | Torch uses TorchANI directly; JAX uses a JAX-native ANI-like bundle interface. |
| `m3gnet` | Torch | [`materialsvirtuallab/matgl`](https://github.com/materialsvirtuallab/matgl) | The Torch backend uses the MatGL-backed M3GNet implementation. |

For MACE specifically, the intended repository split is:

- [`mace-model`](https://github.com/bamescience/mace-model): model definition,
  backend-specific model code, model initialization, conversion, and foundation
  model export
- [`equitrain`](https://github.com/bamescience/equitrain): preprocessing,
  training, fine-tuning, checkpoint handling, and experiment orchestration

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
  * `[torch]` - Install the core Torch backend (PyTorch, torch_geometric, accelerate, torch-ema)
  * `[jax]` - Install the JAX backend runtime (jax, jaxlib)
  * `[ani]` - Install TorchANI models for molecular systems
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

If you are working with MACE, treat
[`mace-model`](https://github.com/bamescience/mace-model) as the companion
model repository and `equitrain` as the training repository. In other words:

1. prepare or convert the model artifact with `mace-model`
2. train or fine-tune it with `equitrain`

The dependency examples below still reflect the current MACE runtime extras in
`equitrain`; the repository boundary above describes the intended long-term
split between model code and training code.

Many examples below use the Torch backend. Ensure the relevant extras are installed, for example:

```bash
pip install equitrain[torch,mace]
```

For JAX-based workflows, install the corresponding extras, e.g.:

```bash
pip install equitrain[jax,mace-jax]
```

ANI can be used through either backend, but the model artifact format differs:

- Torch ANI uses TorchANI models/checkpoints directly and requires the `ani` extra.
- JAX ANI uses a JAX-native ANI-like module packaged as a JAX bundle (`config.json` + `params.msgpack`).

For Torch ANI support:

```bash
pip install equitrain[torch,ani]
```

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

The preprocessing command accepts `.xyz`, `.lmdb`/`.aselmdb`, and `.h5` inputs; LMDB datasets are automatically converted to the native HDF5 format before statistics are computed. XYZ files are parsed through ASE so that lattice vectors, species labels, and per-configuration metadata are retained. Precomputed statistics (means, standard deviations, cutoff radius, atomic energies) are stored alongside and reused by the training entry points.

Under the hood, each processed file is organised as:

- `/structures`: per-configuration metadata (cell, energy, stress, weights, etc.) and pointers into the per-atom arrays.
- `/positions`, `/forces`, `/atomic_numbers`: flat, chunked arrays sized by the total number of atoms across the dataset. Random reads only touch the slices required for a batch.

This layout keeps the HDF5 file compact even for tens of millions of structures: chunked per-atom arrays avoid the pointer-chasing overhead of variable-length fields, enabling efficient multi-worker dataloaders that issue many small reads concurrently.

<!-- TODO: change this following a notebook style -->
#### Python Script:

```python
from equitrain import get_args_parser_preprocess, preprocess


def run_preprocess():
    args = get_args_parser_preprocess().parse_args([])
    args.train_file = 'data.xyz'
    args.valid_file = 'data.xyz'
    args.output_dir = 'test_preprocess'
    args.compute_statistics = True
    args.atomic_energies = 'average'
    args.r_max = 4.5

    preprocess(args)


if __name__ == '__main__':
    run_preprocess()
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
    --model path/to/mace.model \
    --model-wrapper 'mace' \
    --epochs 10 \
    --tqdm

# Training with ORB
equitrain -v \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir result_orb \
    --model path/to/orb.model \
    --model-wrapper 'orb' \
    --epochs 10 \
    --tqdm

# Training with TorchANI
equitrain -v \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir result_ani \
    --model path/to/ani.model \
    --model-wrapper ani \
    --epochs 10 \
    --tqdm

# JAX multi-GPU (single node, auto spawns one process per visible GPU)
CUDA_VISIBLE_DEVICES=0,1 \
equitrain -v \
    --backend jax \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir result_jax \
    --model path/to/mace.model \
    --model-wrapper mace \
    --batch-max-edges 200000 \
    --device gpu \
    --launcher auto \
    --distributed \
    --epochs 10 \
    --tqdm
```

HDF5 inputs can be a directory, a glob (e.g. `data/train_*.h5`), or a comma-separated
list of files; all shards are concatenated in order. This applies to
`--train-file`, `--valid-file`, and `--test-file` when training with either backend.

<!-- TODO: change this following a notebook style -->
#### Python Script:

```python
from equitrain import get_args_parser_train, train


def train_mace():
    args = get_args_parser_train().parse_args([])
    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'
    args.output_dir = 'runs/mace'
    args.epochs = 10
    args.batch_size = 64
    args.lr = 1e-2
    args.verbose = 1
    args.tqdm = True

    args.model = 'path/to/mace.model'
    args.model_wrapper = 'mace'

    train(args)


def train_orb():
    args = get_args_parser_train().parse_args([])
    args.train_file = 'data/train.h5'
    args.valid_file = 'data/valid.h5'
    args.output_dir = 'runs/orb'
    args.epochs = 10
    args.batch_size = 32
    args.lr = 5e-4
    args.verbose = 1
    args.tqdm = True

    args.model = 'path/to/orb.model'
    args.model_wrapper = 'orb'

    train(args)


if __name__ == '__main__':
    train_mace()
    # train_orb()
```

#### Running the JAX backend

The training CLI automatically selects the Torch backend. To run the JAX backend instead, point `--backend` to `jax` and provide a JAX bundle. A JAX bundle is a model directory with a `config.json` file and a `params.msgpack` file:

```text
path/to/jax_bundle/
  config.json
  params.msgpack
```

For MACE, this bundle can be exported via `mace_jax_from_torch` or the fine-tuning utilities:

```bash
equitrain -v \
    --backend jax \
    --model path/to/jax_bundle \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir result-jax \
    --epochs 5
```

For JAX ANI, the bundle must describe how to construct a JAX-native ANI-like module. It does not load TorchANI checkpoints directly. A minimal `config.json` looks like:

```json
{
  "wrapper_name": "ani",
  "atomic_numbers": [1, 6, 7, 8],
  "species_order": ["H", "C", "N", "O"],
  "r_max": 5.2,
  "module_factory": "my_package.my_ani:create_model",
  "model_kwargs": {}
}
```

The `module_factory`, `module_builder`, or `module_class` entry must resolve to a Python object that builds the JAX model. The resulting module should expose an `apply` method that accepts either a mapping with `species`, `coordinates`, `atom_mask`, and `counts`, or positional `(species, coordinates)` inputs. The output must include `energy` and may optionally include `forces` and `stress`. If the module returns only energy and `--forces-weight` is positive, the JAX ANI wrapper computes forces with `jax.grad`.

Example JAX ANI training command:

```bash
equitrain -v \
    --backend jax \
    --model path/to/jax_ani_bundle \
    --model-wrapper ani \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir result-jax-ani \
    --energy-weight 1.0 \
    --forces-weight 1.0 \
    --stress-weight 0.0 \
    --batch-max-edges 10000 \
    --epochs 5
```

Start by testing JAX ANI with `--forces-weight 0.0` for an energy-only smoke test. After the bundle loads and energy training works, enable force training to exercise the `jax.grad` force path.

---

### 3. Making Predictions

Use a trained model to make predictions on new data:

<!-- TODO: change this following a notebook style -->
#### Python Script:

```python
from equitrain import get_args_parser_predict, predict


def predict_with_mace():
    args = get_args_parser_predict().parse_args([])
    args.predict_file = 'data/valid.h5'
    args.batch_size = 64
    args.model = 'path/to/mace.model'
    args.model_wrapper = 'mace'

    energy_pred, forces_pred, stress_pred = predict(args)
    print(energy_pred)
    print(forces_pred)
    print(stress_pred)


if __name__ == '__main__':
    predict_with_mace()
```

For HDF5 inputs you can pass a directory, glob, or comma-separated list of files
via `--predict-file` (all shards are concatenated in order).

The same backend/model distinction applies for prediction. Torch ANI uses a TorchANI checkpoint:

```bash
equitrain-predict \
    --model path/to/ani.model \
    --model-wrapper ani \
    --predict-file data/valid.h5
```

JAX ANI uses the JAX bundle described above:

```bash
equitrain-predict \
    --backend jax \
    --model path/to/jax_ani_bundle \
    --model-wrapper ani \
    --predict-file data/valid.h5 \
    --batch-max-edges 10000
```

---

### 4. ASE Calculators and Relaxation

`equitrain` also exposes a small calculator API for structure-level inference
and geometry optimization with ASE:

Torch:
- `equitrain.calculators.TorchWrapperPredictor`
- `equitrain.calculators.build_ase_calculator`

JAX:
- `equitrain.calculators.JaxWrapperPredictor`
- `equitrain.calculators.build_jax_ase_calculator`

Important behavior:

- `model_wrapper` is required and must be explicit.
- Torch `model` must be either:
  - a loaded `torch.nn.Module`, or
  - an existing model file path.
- JAX `model` must be either:
  - a loaded JAX `ModelBundle` (`config`, `params`, `module`), or
  - an existing JAX bundle path (`config.json` + `params.msgpack`).
- Foundation-model alias resolution is intentionally not done inside the
  calculator; resolve aliases before creating the calculator and pass the loaded
  model (or resolved file path).
- The ASE calculator returns `energy` and `forces` (no stress).
- If a requested GPU/CUDA device is unavailable, the API falls back to CPU.

Supported wrappers:
- Torch calculator: `mace`, `ani`, `orb`, `sevennet`, `m3gnet`
- JAX calculator: wrappers available in `equitrain.backends.jax_wrappers` (currently `mace`, `ani`)

#### Batched Structure Prediction

```python
from ase.build import molecule
from equitrain.calculators import TorchWrapperPredictor

predictor = TorchWrapperPredictor(
    model="path/to/model.pt",
    model_wrapper="mace",
    device="cuda:0",
    default_dtype="float32",
    batch_size=16,
    require_forces=True,
)

atoms = molecule("H2O")
energies, forces = predictor.predict([atoms], require_forces=True)
print(energies[0], forces[0].shape)
```

#### ASE Geometry Optimization

```python
from ase.build import molecule
from ase.optimize import FIRE
from equitrain.calculators import build_ase_calculator

atoms = molecule("H2O")
atoms.calc = build_ase_calculator(
    model="path/to/model.pt",
    model_wrapper="mace",
    device="cuda:0",
    default_dtype="float64",
    batch_size=8,
)

opt = FIRE(atoms, logfile=None)
opt.run(fmax=0.05, steps=200)
print("Relaxed energy:", atoms.get_potential_energy())
```

You can also import these from the top-level package:

```python
from equitrain import build_ase_calculator, get_torch_wrapper_predictor
```

JAX example:

```python
from ase.build import molecule
from equitrain.calculators import build_jax_ase_calculator

atoms = molecule("H2O")
atoms.calc = build_jax_ase_calculator(
    model="path/to/jax_bundle",
    model_wrapper="ani",
    device="cpu",
)
print(atoms.get_potential_energy())
```

---

### JAX Backend Multi-Device Notes

- When the JAX backend detects more than one local accelerator, it automatically switches to a multi-device (`pmap`) execution. In that mode the training and evaluation batch size must be divisible by `jax.local_device_count()` so that each device processes an identical number of graphs.
- On single-device machines no extra configuration is required; the backend falls back to the same single-device behaviour that existing scripts expect.

---

### Fine-Tuning Adapters

Equitrain ships adapter-style fine-tuning helpers in `equitrain.finetune`. They
are currently exposed through the Python API and are designed to keep the
original model frozen while training only a small set of additional parameters.

#### Delta Fine-Tuning

Delta fine-tuning is the simplest adapter method in the repository:

- every selected parameter gets a trainable additive residual with the same shape
- the forward pass uses `base_parameter + delta`
- the base model stays frozen throughout optimisation

This is effectively LoRA without any rank compression. It is useful when you
want the simplest possible residual fine-tuning scheme and do not need to limit
adapter size aggressively.

Implementation details:

- Torch: `DeltaFineTuneWrapper` mirrors all base parameters with same-shaped
  delta tensors and merges them only for the forward pass or export.
- JAX/NNX: `wrap_jax_module_with_deltas()` / `JaxDeltaFineTuneModule` keep the
  frozen model state under `base_params` and the trainable residuals under
  `params.delta`.

Minimal Torch example:

```python
from equitrain import get_args_parser_train, train
from equitrain.finetune import TorchDeltaFineTuneWrapper
from equitrain.utility_test import MaceWrapper

args = get_args_parser_train().parse_args([])
args.train_file = 'data/train.h5'
args.valid_file = 'data/valid.h5'
args.output_dir = 'runs/mace-delta'

base_model = MaceWrapper(args, filename_model='path/to/mace.model')
args.model = TorchDeltaFineTuneWrapper(base_model)

train(args)
```

#### LoRA Fine-Tuning

Equitrain also provides LoRA adapters for both backends:

- Torch: `TorchLoRAFineTuneWrapper`
- JAX/NNX: `wrap_jax_module_with_lora()` / `JaxLoRAFineTuneModule`

The LoRA implementation in this repository is intentionally close to the delta
wrapper, but it only applies low-rank updates to eligible weight tensors:

- only parameters named `*.weight` with `ndim >= 2` receive LoRA adapters
- higher-order weights are flattened to `(shape[0], prod(shape[1:]))`, updated
  as a matrix, and reshaped back to the original tensor shape
- biases and 1D weights remain frozen

Instead of requiring one fixed rank for the whole model, Equitrain lets you
specify either:

- `rank_reduction`: percentage of rank to remove
- `rank_fraction`: percentage of rank to keep

This is usually more practical for MLIPs, because different layers can have very
different matrix sizes. For example, `rank_reduction=75` keeps roughly 25% of
the effective rank of each eligible weight matrix, with a minimum rank of 1.

The effective update is:

```text
W_eff = W + scale * (B @ A)
```

where:

- `A` has shape `(r, in_dim)`
- `B` has shape `(out_dim, r)`
- `scale = alpha / r` if `alpha` is provided, otherwise `scale = 1`

Torch example:

```python
from equitrain import get_args_parser_train, train
from equitrain.finetune import TorchLoRAFineTuneWrapper
from equitrain.utility_test import MaceWrapper

args = get_args_parser_train().parse_args([])
args.train_file = 'data/train.h5'
args.valid_file = 'data/valid.h5'
args.output_dir = 'runs/mace-lora'

base_model = MaceWrapper(args, filename_model='path/to/mace.model')
args.model = TorchLoRAFineTuneWrapper(
    base_model,
    rank_reduction=75,
    alpha=16,
)

train(args)
```

JAX helper example:

```python
from equitrain.finetune import wrap_jax_module_with_lora

lora_module = wrap_jax_module_with_lora(
    jax_module,
    rank_reduction=75,
    alpha=16,
)
variables = lora_module.init()
```

For JAX, the wrapped variable tree stores the frozen imported state under
`base_params` and the trainable LoRA weights under `params.lora`.

---

## Advanced Features

### Multi-GPU and Multi-Node Training

Equitrain supports multi-GPU and multi-node training using `accelerate`. Example scripts are available in the `resources/training` directory.

### Dataset Preparation

Equitrain provides scripts for downloading and preparing popular datasets such as Alexandria and MPTraj. These scripts can be found in the `resources/data` directory.

### Pretrained Models

Initial model examples and configurations can be accessed in the `resources/models` directory.
