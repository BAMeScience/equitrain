# Fine-Tuning

Fine-tuning helpers live in `equitrain.finetune`. They keep the base model
frozen and train a smaller set of additional or selected parameters.

Use a non-zero `--weight-decay`/`args.weight_decay` and tune it on validation
data. Values such as `1e-6` to `1e-4` are typical starting points, depending on
dataset size and adapter capacity.

Fine-tuning uses the normal training loop: it runs for `--epochs`; there is no
separate early-stopping criterion. Equitrain evaluates before training and
after every epoch, logs to `trainer.log`, and saves a best checkpoint whenever
validation total loss improves. By default all best checkpoints are kept; set
`--keep-best-checkpoints N` to retain only the N checkpoints with the lowest
validation total loss. Use the best validation checkpoint rather than assuming
the final epoch is best.

Convergence is usually checked from the validation loss curve and from
checkpoint names such as `best_val_epochs@<epoch>_e@<loss>`. If validation loss
flattens or starts increasing while training loss still decreases, the run has
stopped improving or is beginning to overfit.

## Delta / L<sup>2</sup>-SP

Delta fine-tuning gives every selected base parameter a same-shaped trainable
residual and evaluates `base_parameter + delta` while the base model remains
frozen. This is Equitrain's residual-parameter implementation of
L<sup>2</sup>-SP ("Starting Point") regularization from Li, Grandvalet, and
Davoine, 2018,
[*Explicit Inductive Bias for Transfer Learning with Convolutional Networks*](https://proceedings.mlr.press/v80/li18a.html).

L<sup>2</sup>-SP regularizes fine-tuned parameters toward their pre-trained
starting values:

```text
Omega(theta) = lambda / 2 * ||theta - theta_0||_2^2
```

Equitrain parameterizes this as `theta = theta_0 + delta`, so weight decay on
trainable deltas regularizes `||delta||_2^2`. The base parameter `theta_0` is
frozen and each delta is initialized at zero. Compared with LoRA, delta
fine-tuning uses full-size residuals rather than low-rank residuals, so it is
useful when you want the simplest residual scheme and do not need to limit
adapter size aggressively.

Implementation details:

- Torch: `DeltaFineTuneWrapper` mirrors selected base parameters with
  same-shaped delta tensors and merges them only for the forward pass or export.
- JAX/NNX: `wrap_jax_module_with_deltas()` / `JaxDeltaFineTuneModule` keep the
  frozen model state under `base_params` and the trainable residuals under
  `params.delta`.

Torch adapter constructors wrap an Equitrain Torch model wrapper, such as a
`MaceWrapper`, `AniWrapper`, or `OrbWrapper` instance:

```python
from equitrain.finetune import TorchDeltaFineTuneWrapper

args.model = TorchDeltaFineTuneWrapper(base_wrapper)
```

Minimal Torch training shape:

```python
from equitrain import get_args_parser_train, train
from equitrain.finetune import TorchDeltaFineTuneWrapper

args = get_args_parser_train().parse_args([])
args.train_file = 'data/train.h5'
args.valid_file = 'data/valid.h5'
args.output_dir = 'runs/mace-delta'
args.weight_decay = 1e-6
args.model = TorchDeltaFineTuneWrapper(base_wrapper)

train(args)
```

JAX/NNX:

```python
from equitrain.finetune import wrap_jax_module_with_deltas

jax_module = wrap_jax_module_with_deltas(jax_module)
variables = jax_module.init()
```

For Torch MACE models, semantic delta layers are ordered as:

```text
0: node_embedding
1: interactions.0
2: interactions.1
3: products.0
4: products.1
5: readouts
```

Passing `freeze_layers="2-"` keeps only the node embedding and first
interaction block trainable.

Delta plus `freeze_layers` is targeted L<sup>2</sup>-SP
(L<sup>2</sup>-TSP): the L<sup>2</sup>-SP penalty is applied only to selected
trainable delta layers, while frozen layers keep `delta = 0` and remain exactly
at their pre-trained starting values.

## Freeze

`TorchFreezeFineTuneWrapper` uses the same semantic layer selection interface
without adapter tensors. It freezes selected base layers and trains the
remaining base weights directly, so exported models already contain the
fine-tuned weights and do not require a delta merge.

```python
from equitrain.finetune import TorchFreezeFineTuneWrapper

args.model = TorchFreezeFineTuneWrapper(base_wrapper, freeze_layers='2-')
```

For MACE, the layer order is the same as delta fine-tuning. Thus
`freeze_layers="2-"` keeps the node embedding and first interaction block
trainable and freezes later blocks.

## LoRA

LoRA adapters are available for Torch and JAX/NNX:

- Torch: `TorchLoRAFineTuneWrapper`
- JAX/NNX: `wrap_jax_module_with_lora()` / `JaxLoRAFineTuneModule`

Equitrain applies LoRA only to eligible `*.weight` tensors with `ndim >= 2`.
Higher-order weights are flattened to matrices for the update and reshaped back
to their original tensor shape. Biases and 1D weights remain frozen.

Use `rank_reduction` to specify the percentage of rank to remove, or
`rank_fraction` to specify the percentage to keep. The effective update is:

```text
W_eff = W + scale * (B @ A)
```

where `A` has shape `(r, in_dim)`, `B` has shape `(out_dim, r)`, and
`scale = alpha / r` when `alpha` is provided, otherwise `scale = 1`.
For example, `rank_reduction=75` keeps roughly 25% of the effective rank of
each eligible weight matrix, with a minimum rank of 1.

```python
from equitrain.finetune import TorchLoRAFineTuneWrapper

args.model = TorchLoRAFineTuneWrapper(
    base_wrapper,
    rank_reduction=75,
    alpha=16,
)
```

JAX helper:

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
`base_params` and trainable LoRA weights under `params.lora`.

## Exporting Torch Fine-Tuned Checkpoints

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

If adapter metadata is unavailable, pass `--fine-tune-wrapper delta`, `lora`, or
`freeze` explicitly.

The checkpoint must have been created with current Equitrain so its `args.json`
contains adapter export metadata for automatic detection. Training checkpoint
directories contain optimizer/training state and are not the same artifact type
as a full exported model file.
