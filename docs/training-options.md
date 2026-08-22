# Training Options

This page summarizes the training controls that are shared by the CLI and the
Python API. The same fields are available on the `argparse.Namespace` returned
by `get_args_parser_train()`.

## Targets and Losses

Equitrain trains a weighted sum of enabled target losses:

```text
total = energy_weight * energy_loss
      + forces_weight * forces_loss
      + stress_weight * stress_loss
```

Torch can also add reaction-relative terms:

```text
total += barrier_weight * barrier_loss
total += reaction_energy_weight * reaction_energy_loss
```

At least one loss weight must be non-zero. JAX currently supports energy,
forces, and stress losses; reaction-relative losses are Torch-only.

Select the base loss with:

- `--loss-type mae`
- `--loss-type smooth-l1`
- `--loss-type mse`
- `--loss-type huber`

Override individual targets with `--loss-type-energy`,
`--loss-type-forces`, and `--loss-type-stress`. The Smooth-L1 and Huber
parameters are controlled by `--smooth-l1-beta` and `--huber-delta`.

`--loss-monitor` is a comma-separated list of additional metrics to log, for
example `--loss-monitor mae,mse`. The main loss type is removed from this list
so it is not logged twice.

`--loss-clipping VALUE` clips per-sample/per-component loss values before they
are averaged. This can reduce the impact of extreme outliers.

## Loss Weighting

`--loss-weight-type groundstate` applies an additional target-dependent factor:

```text
weight = exp(-1000 * ||target||^2) + 1
```

It can be set globally or per target with `--loss-weight-type-energy`,
`--loss-weight-type-forces`, and `--loss-weight-type-stress`.

The Equitrain HDF5 format also stores per-configuration target availability
weights such as `energy_weight`, `forces_weight`, and `stress_weight`. Missing
targets are written with zero target weights during preprocessing; global CLI
loss weights still control which losses are requested.

## Optimizers

`--opt` selects the Torch optimizer. Supported names are:

- `adamw`
- `adam`
- `sgd`
- `nesterov`
- `momentum`
- `adadelta`
- `rmsprop`

Common optimizer options are:

- `--lr`: initial learning rate.
- `--weight-decay`: weight decay for trainable non-bias parameters.
- `--gradient-clipping`: gradient clipping threshold.
- `--momentum`: used by SGD, Nesterov, Momentum, and RMSprop.
- `--alpha`: RMSprop smoothing constant.

JAX supports `adamw`, `adam`, `sgd`, `nesterov`, `momentum`, and `rmsprop`
through Optax. `adadelta` is Torch-only.

## Schedulers

Torch supports `--scheduler plateau`, `step`, and `exponential`. JAX supports
those names and also treats `none` or `constant` as a fixed learning rate.

- `plateau`: reduce the learning rate when the monitored loss stops improving.
- `step`: reduce the learning rate every `--step-size` epochs by `--gamma`.
- `exponential`: reduce the learning rate every epoch by `--gamma`.

For plateau scheduling, use `--plateau-factor`, `--plateau-patience`,
`--plateau-threshold`, `--plateau-threshold-mode`, `--plateau-mode`,
`--plateau-eps`, and `--min-lr`.

`--scheduler_monitor train` uses training total loss for the scheduler.
`--scheduler_monitor val` uses validation total loss.

## EMA

Enable exponential moving average parameters with:

```bash
equitrain ... --ema --ema-decay 0.999
```

Torch checkpoints write EMA state to `ema.bin` when EMA is enabled. JAX
checkpoints write `ema_params.msgpack`.

## Checkpoints

Equitrain evaluates before training and after every epoch. A new best
validation checkpoint is saved whenever validation total loss improves.
Checkpoint directories are named:

```text
best_val_epochs@<epoch>_e@<loss>
```

By default all best checkpoints are kept. Limit retention with:

```bash
equitrain ... --keep-best-checkpoints 3
```

Resume a full training state, including optimizer and scheduler state:

```bash
equitrain ... --load-best-checkpoint
equitrain ... --load-last-checkpoint
equitrain ... --load-checkpoint runs/job/best_val_epochs@10_e@0.0123
```

Load only model weights and start a new optimizer/scheduler state:

```bash
equitrain ... --load-best-checkpoint-model
equitrain ... --load-last-checkpoint-model
equitrain ... --load-checkpoint-model runs/job/best_val_epochs@10_e@0.0123/pytorch_model.bin
```

For JAX checkpoints, model weights are stored as `params.msgpack`. Torch
checkpoints usually contain `pytorch_model.bin` or `model.safetensors`.

Inspect a model or checkpoint-loaded model:

```bash
equitrain-inspect \
    --model path/to/model.pt \
    --model-wrapper mace

equitrain-inspect \
    --model path/to/model.pt \
    --model-wrapper mace \
    --output-dir runs/job \
    --load-best-checkpoint-model
```

Export a Torch model file from a checkpoint with `equitrain-export`. Fine-tuned
Torch adapter checkpoints are reconstructed and merged automatically when the
checkpoint contains current adapter metadata. See
[Fine-Tuning](fine-tuning.md#exporting-torch-fine-tuned-checkpoints).

## Parameter Freezing

`--freeze-params` and `--unfreeze-params` accept Python regular expressions
matched with `re.fullmatch` against parameter names.

With `--freeze-params`, matching parameters are frozen and all others keep their
current trainability. With `--unfreeze-params`, matching parameters are trainable
and all others are frozen.

Torch applies this through `requires_grad`. JAX builds a matching optimizer mask.
This is lower-level than the Delta, Freeze, and LoRA wrappers described in
[Fine-Tuning](fine-tuning.md).

## Batching and Runtime

Torch workflows normally use `--batch-size`. Optional graph caps
`--batch-max-nodes` and `--batch-max-edges` split large collated batches; with
`--batch-drop`, individual graphs exceeding the cap are dropped.

JAX train/evaluate workflows require `--batch-max-edges`; the loader infers
fixed padded shapes from the data and edge cap. JAX prediction also accepts
`--batch-max-nodes` and can reuse cached streaming statistics when explicit caps
are omitted. The JAX loader streams fixed-shape padded batches from HDF5 and can
prefetch batches with `--prefetch-batches`.

`--num-workers` and `--pin-memory` affect data loading. For Torch, worker
processes and pinned memory are only enabled when they are useful for the active
Accelerate device. For JAX, workers build graph batches from HDF5 directly.

Use `--train-max-steps` and `--valid-max-steps` for short smoke tests or bounded
debug runs.

## Weighted Sampler

Torch training can resample configurations according to their current
prediction error:

```bash
equitrain ... --weighted-sampler
```

`--weighted-sampler-threshold VALUE` suppresses errors above the threshold when
building sampler weights, then replaces zero entries with the mean error.

The weighted sampler is Torch-only and is incompatible with reaction-relative
losses.
