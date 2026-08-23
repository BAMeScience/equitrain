# Phonon Fine-Tuning Paper

This page describes how to reproduce the Equitrain fine-tuning setup used in
the manuscript *Data-Efficient Fine-Tuning of Machine-Learning Interatomic
Potentials for Phonon and Thermal Properties* with current Equitrain.

The manuscript used the legacy Equitrain 0.1.0 Delta wrapper. Current Equitrain
keeps the same L<sup>2</sup>-SP objective available, but the layer selection is
now explicit.

## Method

The paper fine-tunes one MACE-MP-0b3 model per material. The Equitrain method is
targeted L<sup>2</sup>-SP, abbreviated `L2-TSP` in the paper and rendered here
as L<sup>2</sup>-TSP:

- the pre-trained MACE parameters stay frozen as `theta_0`;
- trainable residual parameters `delta` are initialized at zero;
- the forward pass evaluates `theta = theta_0 + delta`;
- AdamW weight decay on `delta` penalizes `||delta||_2^2`, which is equivalent
  to the L<sup>2</sup>-SP penalty `||theta - theta_0||_2^2`;
- only the node embedding and first interaction layer are adapted.

In Equitrain 0.1.0, the old Delta wrapper effectively exposed only the early
MACE deltas to the optimizer for this model. Current Equitrain does not rely on
that implicit behavior. To reproduce the paper setup, pass
`freeze_layers="2-"` to `TorchDeltaFineTuneWrapper`. This freezes semantic layer
indices 2 and above, leaving indices 0 and 1 trainable.

For MACE models with two interaction/product blocks, current Equitrain orders
the semantic layers as:

```text
0: node_embedding
1: interactions.0
2: products.0
3: interactions.1
4: products.1
5: readouts
```

Thus `freeze_layers="2-"` leaves the paper's targeted layers, `node_embedding`
and `interactions.0`, trainable.

## Data Setup

The paper constructs material-specific fine-tuning data around equilibrium:

- generate primitive, small-supercell, or large-supercell rattled structures;
- use cubic small supercells between `(5 A)^3` and `(10 A)^3`, and large
  supercells between `(10 A)^3` and `(15 A)^3`;
- create five rattled configurations with a mode amplitude of about `0.1 A`;
- perturb lattice angles and vectors within `+-2%`;
- use the five volume-scaled trajectories `-5%`, `-2.5%`, `0%`, `+2.5%`, and
  `+5%`;
- relax the rattled structures with MACE-MP-0b3 until forces converge to
  `1e-4 eV/A`;
- select four configurations per relaxation trajectory, equally spaced in
  energy from the starting structure to 90% convergence, for DFT labels;
- generate 20 labeled configurations per material: the four configurations from
  the `0%` trajectory are validation data, and the remaining 16 are training
  data;
- for the main phonon comparisons, use 10 large-supercell training structures
  per material.

Current Equitrain training expects the current HDF5 layout documented in
[Data and Preprocessing](data.md). If you start from XYZ or another ASE-readable
format, preprocess the train/validation structures first:

```bash
equitrain-preprocess \
    --train-file material-train.xyz \
    --valid-file material-valid.xyz \
    --compute-statistics \
    --atomic-energies average \
    --r-max 6.0 \
    --output-dir data/material
```

## Paper Hyperparameters

| Setting | Value |
| --- | --- |
| Foundation model | MACE-MP-0b3 |
| Fine-tuning strategy | targeted L<sup>2</sup>-SP / `L2-TSP` |
| Trainable semantic layers | `node_embedding`, `interactions.0` |
| Cutoff radius | `6.0 A` |
| Optimizer | AdamW |
| Scheduler | ReduceLROnPlateau |
| Epochs | up to `200` |
| Learning rate | `0.01` |
| Weight decay | `10.0` for Equitrain L<sup>2</sup>-SP deltas |
| Loss | Huber with `delta = 0.01` |
| Loss weights | energy `10`, forces `100`, stress `1000` |
| EMA | disabled |

## Current Equitrain Reproduction

Construct the fine-tuned model from Python so the adapter is created before the
normal training loop starts:

```python
from equitrain import get_args_parser_train, get_model, train
from equitrain.finetune import TorchDeltaFineTuneWrapper

args = get_args_parser_train().parse_args([])
args.backend = "torch"
args.train_file = "data/material/train.h5"
args.valid_file = "data/material/valid.h5"
args.output_dir = "runs/material-l2-tsp"
args.model = "path/to/mace-mp-0b3.model"
args.model_wrapper = "mace"
args.dtype = "float32"

args.r_max = 6.0
args.loss_type = "huber"
args.huber_delta = 0.01
args.energy_weight = 10.0
args.forces_weight = 100.0
args.stress_weight = 1000.0
args.opt = "adamw"
args.lr = 0.01
args.weight_decay = 10.0
args.scheduler = "plateau"
args.epochs = 200
args.ema = False

base_model = get_model(args)
args.model = TorchDeltaFineTuneWrapper(base_model, freeze_layers="2-")

train(args)
```

Do not omit `freeze_layers="2-"` when reproducing the paper. Without it,
current Equitrain trains deltas for all semantic layers, which is standard
L<sup>2</sup>-SP rather than the targeted L<sup>2</sup>-TSP setup used in the
paper.

## Export

Export the best checkpoint before using the model for prediction, calculators,
or phonon workflows:

```bash
equitrain-export -v \
    --model path/to/mace-mp-0b3.model \
    --model-wrapper mace \
    --output-dir runs/material-l2-tsp \
    --load-best-checkpoint \
    --model-export runs/material-l2-tsp/mace-l2-tsp.model
```

Current checkpoints store fine-tuning metadata, so export normally detects the
Delta wrapper and `freeze_layers="2-"` automatically. If the metadata is
missing, add `--fine-tune-wrapper delta`.

## Regression Coverage

Equitrain includes regression tests that compare the current targeted
L<sup>2</sup>-SP configuration against golden values generated with Equitrain
0.1.0. These tests protect both the old layer-selection behavior and a simple
optimizer-step trajectory for the Delta wrapper.
