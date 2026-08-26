# Python API

The CLI and Python API share the same `argparse.Namespace` objects. Start from
the same parser that the CLI uses, modify fields programmatically, then call
the workflow function.

Available parser constructors are:

- `get_args_parser_preprocess()`
- `get_args_parser_train()`
- `get_args_parser_evaluate()`
- `get_args_parser_predict()`
- `get_args_parser_inspect()`
- `get_args_parser_export()`

In notebooks and tests, use `parse_args([])` to avoid inheriting unrelated
process arguments.

Generated entries for stable public objects are collected in the
[API Reference](api-reference.md).

## Preprocess

```python
from equitrain import get_args_parser_preprocess, preprocess

args = get_args_parser_preprocess().parse_args([])
args.train_file = 'data-train.xyz'
args.valid_file = 'data-valid.xyz'
args.output_dir = 'data'
args.compute_statistics = True
args.atomic_energies = 'average'
args.r_max = 4.5

preprocess(args)
```

## Train

```python
from equitrain import get_args_parser_train, train

args = get_args_parser_train().parse_args([])
args.train_file = 'data/train.h5'
args.valid_file = 'data/valid.h5'
args.output_dir = 'runs/mace'
args.model = 'path/to/mace.model'
args.model_wrapper = 'mace'
args.epochs = 10
args.batch_size = 64
args.verbose = 1
args.tqdm = True

train(args)
```

For JAX:

```python
args.backend = 'jax'
args.model = 'path/to/jax_bundle'
args.model_wrapper = 'mace'
args.batch_max_edges = 200000
```

For ORB, use the same parser and switch wrapper/model/output fields:

```python
args.model = 'path/to/orb.model'
args.model_wrapper = 'orb'
args.output_dir = 'runs/orb'
args.lr = 5e-4
```

## Model Loading

Use `get_model(args)` when you need the backend-specific model object without
starting a training/evaluation/prediction workflow:

```python
from equitrain import get_args_parser_train, get_model

args = get_args_parser_train().parse_args([])
args.model = 'path/to/mace.model'
args.model_wrapper = 'mace'

model = get_model(args)
```

With `args.backend = 'jax'`, `get_model(args)` returns a JAX `ModelBundle`
loaded from the bundle directory.

## Checkpoint Helpers

Backend-aware checkpoint helpers live in `equitrain.checkpoint`:

```python
from equitrain.checkpoint import load_checkpoint, load_model_state, save_checkpoint
```

Most workflows should prefer the CLI/Python workflow arguments documented in
[Training Options](training-options.md#checkpoints). Use these helpers when you
are integrating Equitrain checkpoint loading into custom training code.

## Evaluate

```python
from equitrain import evaluate, get_args_parser_evaluate

args = get_args_parser_evaluate().parse_args([])
args.test_file = 'data/test.h5'
args.model = 'path/to/mace.model'
args.model_wrapper = 'mace'
args.batch_size = 64
args.output_dir = 'evaluation_mace'

metrics = evaluate(args)
```

## Predict

```python
from equitrain import get_args_parser_predict, predict

args = get_args_parser_predict().parse_args([])
args.predict_file = 'data/valid.h5'
args.model = 'path/to/mace.model'
args.model_wrapper = 'mace'
args.batch_size = 64
args.output_dir = 'predictions_mace'

energy_pred, forces_pred, stress_pred = predict(args)
```

When `output_dir` is set, predictions are also written to `predictions.npz` and
metadata is written to `predictions.json`.

## Structure-Level Prediction

For most ASE workflows, prefer the calculator APIs in
[Calculators](calculators.md). They load the wrapped model, build graphs, and
return energies/forces through an ASE-compatible interface.

The lower-level Torch helpers `predict_atoms` and `predict_structures` are also
available when you already have a loaded Torch wrapper, an `AtomicNumberTable`,
and a cutoff radius. Use `predict_graphs` when you already have Torch graph
batches:

```python
from ase.build import molecule
from equitrain import predict_atoms
from equitrain.data.atomic import AtomicNumberTable

atoms = [molecule('H2O')]
z_table = AtomicNumberTable([1, 8])
energy, forces, stress = predict_atoms(
    model,
    atoms,
    z_table,
    r_max=4.5,
    batch_size=16,
)
```

## Data Helpers

The stable data helpers are re-exported from `equitrain.data`:

```python
from equitrain.data import AtomicNumberTable, Configuration, Statistics
from equitrain.data.format_hdf5 import HDF5Dataset, HDF5GraphDataset
from equitrain.data.format_lmdb import convert_lmdb_to_hdf5
```

`HDF5Dataset` reads and writes ASE `Atoms` objects using the layout documented
in [Data and Preprocessing](data.md). `HDF5GraphDataset` adds Torch graph
construction on top of the same file format.
