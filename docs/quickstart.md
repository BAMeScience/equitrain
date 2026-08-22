# Quickstart

This page shows the common command-line workflow: preprocess data, train a
model, evaluate a checkpoint, and run prediction.

## 1. Preprocess

```bash
equitrain-preprocess \
    --train-file data-train.xyz \
    --valid-file data-valid.xyz \
    --compute-statistics \
    --atomic-energies average \
    --output-dir data \
    --r-max 4.5
```

`equitrain-preprocess` accepts `.xyz`, `.lmdb`/`.aselmdb`, and `.h5` inputs.
LMDB datasets are converted to Equitrain HDF5 before statistics are computed.
XYZ files are parsed through ASE so lattice vectors, species labels, and
per-configuration metadata are retained. Precomputed statistics such as means,
standard deviations, cutoff radius, and atomic energies are stored alongside
the output data and reused by training entry points.

The HDF5 layout is:

- `/structures`: per-configuration metadata such as cell, energy, stress,
  weights, charge, spin, external field, and reaction metadata.
- `/positions`, `/forces`, `/atomic_numbers`: flat, chunked per-atom arrays
  addressed by offsets stored in `/structures`.

This layout keeps large HDF5 files compact: random reads only touch the atom
slices required for each batch, avoiding variable-length pointer chasing.

Torch HDF5 paths passed to training, evaluation, or prediction can be a file,
directory, glob, or comma-separated list. Multiple shards are concatenated in
order. JAX CLI workflows expect explicit HDF5 file paths.

MACE-POLAR/PolarMACE system fields `charge`, `spin`, and `external_field` are
preserved as `total_charge`, `total_spin`, and `external_field`. Override XYZ
keys with `--total-charge-key`, `--total-spin-key`, and
`--external-field-key`.

Reaction-relative metadata is also preserved during preprocessing. See
[Data and Preprocessing](data.md) for the full metadata/key contract and
[Reaction-Relative Losses](reaction-relative-losses.md) for reaction-specific
fields.

## 2. Train

Torch/MACE example:

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

Use the same pattern with `--model-wrapper orb`, `ani`, `sevennet`, or
`m3gnet` when the corresponding extra and model artifact are available.

Torch ORB and TorchANI use the same CLI shape:

```bash
equitrain -v \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir runs/orb \
    --model path/to/orb.model \
    --model-wrapper orb \
    --epochs 10 \
    --tqdm

equitrain -v \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir runs/ani \
    --model path/to/ani.model \
    --model-wrapper ani \
    --epochs 10 \
    --tqdm
```

JAX training uses `--backend jax` and a JAX model bundle:

```text
path/to/jax_bundle/
  config.json
  params.msgpack
```

```bash
equitrain -v \
    --backend jax \
    --model path/to/jax_bundle \
    --model-wrapper mace \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir runs/jax-mace \
    --batch-max-edges 200000 \
    --epochs 10 \
    --tqdm
```

JAX ANI and JAX M3GNet bundles must define a JAX-native module factory/class in
`config.json`; they do not load TorchANI or MatGL Torch checkpoints directly.
See [JAX Bundles](jax-bundles.md).

For the full set of loss, optimizer, scheduler, checkpoint, freezing, and
batching controls, see [Training Options](training-options.md).

For single-node multi-GPU JAX training, make the intended devices visible and
let the automatic launcher spawn one process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1 equitrain -v \
    --backend jax \
    --distributed \
    --launcher auto \
    --jax-platform gpu \
    --model path/to/jax_bundle \
    --model-wrapper mace \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --batch-max-edges 200000 \
    --output-dir runs/jax-mace \
    --epochs 10 \
    --tqdm
```

## 3. Evaluate

```bash
equitrain-evaluate -v \
    --test-file data/test.h5 \
    --model path/to/mace.model \
    --model-wrapper mace \
    --batch-size 64 \
    --output-dir evaluation_mace
```

When `--output-dir` is set, evaluation writes `test_metrics.json`. Torch also
writes `test_errors.csv`; JAX currently writes aggregate metrics only.
`test_metrics.json` records the backend, input dataset, primary loss type, and
one metrics block per monitored loss type. Each metric component contains
`avg`, `sum`, and `count`. For Torch, `test_errors.csv` contains one
`index,error` row per configuration. For JAX, `errors_file` is currently
`null`.

If `--output-dir` is omitted, evaluation still logs aggregate metrics and the
Python API returns the metric object, but no files are written.

For JAX evaluation, provide the backend and graph-packing limit:

```bash
equitrain-evaluate -v \
    --backend jax \
    --test-file data/test.h5 \
    --model path/to/jax_bundle \
    --model-wrapper mace \
    --batch-max-edges 200000 \
    --output-dir evaluation_jax
```

## 4. Predict

```bash
equitrain-predict \
    --model path/to/mace.model \
    --model-wrapper mace \
    --predict-file data/valid.h5 \
    --batch-size 64 \
    --output-dir predictions_mace
```

With `--output-dir`, Equitrain writes `predictions.npz` and
`predictions.json`. The NPZ contains available prediction arrays such as
`energy`, `forces`, and `stress`; the JSON records backend, input dataset,
array file name, shapes, and dtypes. Without `--output-dir`, the CLI prints
predictions and the Python API returns arrays.

Torch ANI prediction uses a TorchANI checkpoint:

```bash
equitrain-predict \
    --model path/to/ani.model \
    --model-wrapper ani \
    --predict-file data/valid.h5 \
    --output-dir predictions_ani
```

JAX ANI prediction uses a JAX bundle:

```bash
equitrain-predict \
    --backend jax \
    --model path/to/jax_ani_bundle \
    --model-wrapper ani \
    --predict-file data/valid.h5 \
    --batch-max-edges 10000 \
    --output-dir predictions_jax_ani
```

## 5. Export Fine-Tuned Torch Checkpoints

Fine-tuned Torch checkpoints should be exported before prediction or calculator
use. Adapter checkpoints are merged during export when their metadata is
available:

```bash
equitrain-export -v \
    --model path/to/base-mace.model \
    --model-wrapper mace \
    --output-dir runs/mace-finetune \
    --load-best-checkpoint \
    --model-export runs/mace-finetune/mace-finetuned.model
```

For a specific checkpoint directory, replace `--load-best-checkpoint` with
`--load-checkpoint runs/mace-finetune/best_val_epochs@...`.

Training checkpoint directories contain model weights plus optimizer/training
state; they are not the same artifact type as the full model file expected by
`--model`. Current Equitrain checkpoints include adapter export metadata in
`args.json`, which allows `equitrain-export` to detect and merge supported
adapter checkpoints automatically.
