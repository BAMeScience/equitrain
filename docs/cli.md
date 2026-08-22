# CLI

Equitrain installs focused command-line entry points for each workflow.

| Command | Purpose |
| --- | --- |
| `equitrain` | Train a model. |
| `equitrain-preprocess` | Convert input data to Equitrain HDF5 and compute statistics. |
| `equitrain-evaluate` | Evaluate a model on a test set. |
| `equitrain-predict` | Run prediction on an HDF5 dataset. |
| `equitrain-export` | Export a Torch model checkpoint, including supported fine-tuning adapters. |
| `equitrain-inspect` | Inspect model/checkpoint state. |
| `equitrain-hdf5-info` | Summarize Equitrain HDF5 files. |
| `equitrain-hdf5-benchmark` | Benchmark sequential HDF5 read throughput. |

Every command supports `-h`/`--help`:

```bash
equitrain --help
equitrain-preprocess --help
equitrain-evaluate --help
equitrain-predict --help
equitrain-export --help
```

Use this page as an entry-point map. Detailed data conversion options are in
[Data and Preprocessing](data.md), and training/checkpoint controls are in
[Training Options](training-options.md).

## Common Arguments

Training, evaluation, and prediction share these model arguments:

- `--backend`: `torch` by default, or `jax`.
- `--model`: Torch model file path or JAX bundle path.
- `--model-wrapper`: wrapper name such as `mace`, `ani`, `orb`, `sevennet`, or
  `m3gnet`.
- `--r-max`: optional cutoff override.
- `--dtype`: default numeric dtype.

Torch workflows usually use `--batch-size`. JAX train/evaluate workflows
require `--batch-max-edges` for fixed-shape graph batches; JAX prediction may
also use `--batch-max-nodes`. See
[Training Options](training-options.md#batching-and-runtime).

## Loss Arguments

The standard loss weights are:

- `--energy-weight`
- `--forces-weight`
- `--stress-weight`

The loss function is selected with `--loss-type` and can be overridden per
quantity with `--loss-type-energy`, `--loss-type-forces`, and
`--loss-type-stress`.

Torch also supports reaction-relative losses:

- `--barrier-weight`
- `--reaction-energy-weight`

These are described in [Reaction-Relative Losses](reaction-relative-losses.md).

## JAX Distributed Arguments

JAX training can initialize `jax.distributed`:

```bash
equitrain \
    --backend jax \
    --distributed \
    --launcher none \
    --process-count <global-processes> \
    --process-index <rank> \
    --coordinator-address <host:port> \
    ...
```

On a single node with multiple visible GPUs, the default launcher behavior can
spawn local JAX processes automatically. For multi-node jobs, launch one
Equitrain process per JAX process and pass explicit process metadata.
