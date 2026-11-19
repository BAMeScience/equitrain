# Native GRACE-1L: Torch and JAX

Equitrain includes native PyTorch and JAX implementations of the
`GRACE_1LAYER_v2_25` architecture from the local TensorPotential 0.6.2.1
reference. Select `--backend torch` or `--backend jax` with `--model-wrapper grace`.
TensorFlow and TensorPotential are not runtime dependencies. No pretrained
weights are downloaded.

## Supported scope

- Angular orders `lmax=0..4` and correlation orders `max_order=1..4`.
- Unnormalized first-kind Chebyshev radial basis and polynomial cutoff.
- Chemical embeddings, normalized radial MLP, equivariant channel mixing,
  reference Clebsch–Gordan products and channel pruning, element-specific
  scalar reductions, and linear-plus-MLP energy readout.
- Constant energy scale, constant or per-element reference energies, and
  trainable per-element shifts.
- Batched periodic structures, energy-derived forces and stress, including
  the mixed derivatives required for force/stress training.
- Float32 and float64, ordinary Equitrain losses, checkpoints, and prediction
  interfaces; Torch export and portable JAX checkpoint bundles.
- JAX JIT and higher derivatives, with explicit masking of padded graphs,
  atoms, and bonds. The JAX implementation imports neither Torch nor TensorFlow.

This initial implementation does not load upstream SavedModels, YAML models,
or pretrained checkpoints. GRACE-2L/3L, alternate radial bases, pair-specific
cutoffs, ZBL repulsion, uncertainty, and LAMMPS export are outside its supported
scope. Unknown JSON fields fail instead of being silently ignored. GPU speed
and distributed training have not yet been benchmarked.

## Train from scratch with Torch

Install Equitrain's existing Torch dependencies:

```bash
pip install -e '.[torch]'
```

Copy and edit `resources/models/grace/small.json`. The example supports hydrogen
and oxygen; replace `atomic_numbers` with the elements in your data. Keep
`atomic_energies` in the same explicit element order. Set the cutoff to match
your preprocessing, `avg_n_neigh` to the average neighbor count, and
`energy_scale`/`atomic_energies` to suitable training-data statistics. The example
values are initialization settings, not fitted physical parameters.

```bash
equitrain \
  --backend torch --model-wrapper grace \
  --model resources/models/grace/small.json \
  --train-file data/train.h5 --valid-file data/valid.h5 \
  --output-dir runs/grace --epochs 10 \
  --dtype float64 --energy-weight 1 --forces-weight 1 --stress-weight 0
```

A JSON file constructs a randomly initialized model. To restore trained weights,
use the same JSON architecture with Equitrain's `--load-checkpoint` option.
Export a trained checkpoint into a standalone Torch model:

```bash
equitrain-export \
  --model-wrapper grace --model resources/models/grace/small.json \
  --load-checkpoint runs/grace/best_val_epochs@1_e@VALUE \
  --model-export grace-trained.pt
```

Use the actual checkpoint directory created by your run. Supply
`--model grace-trained.pt --model-wrapper grace` to the usual evaluation or
prediction commands. Loading a JSON alone for evaluation would use fresh random
weights.

For Python construction:

```python
from equitrain.models.grace import Grace1L, GraceConfig
from equitrain.backends.torch_wrappers import GraceWrapper

native = Grace1L(GraceConfig(atomic_numbers=(1, 8), lmax=2)).double()
model = GraceWrapper(args, native)  # args supplies forces_weight/stress_weight
prediction = model(graph_batch)
```

Output shapes are energy `[structures]`, forces `[atoms, 3]`, and stress
`[structures, 3, 3]`. Disabled derivatives are `None`. Stress is the derivative
with respect to symmetric strain divided by cell volume (ASE's tensile-positive
convention); it requires a real, nonzero cell volume. Full periodic bond vectors,
including image shifts, are strained together. Forces are negative position
gradients. Use `model.train()` for differentiable derivative losses and
`model.eval()` for prediction. Derivatives work inside `torch.no_grad()`, but
not `torch.inference_mode()`.

## Train from scratch with JAX

Install the JAX extra, then initialize a bundle from the same architecture JSON
used by Torch. Initialization is explicit: loading a bundle always loads saved
parameters rather than silently generating new weights.

```bash
pip install -e '.[jax]'
python -m equitrain.models.grace.jax_model \
  --config resources/models/grace/small.json \
  --output runs/grace-jax-initial --seed 42 --dtype float64

equitrain \
  --backend jax --model-wrapper grace \
  --model runs/grace-jax-initial \
  --train-file data/train.h5 --valid-file data/valid.h5 \
  --output-dir runs/grace-jax --epochs 10 \
  --batch-max-edges 2048 --num-workers 0 \
  --dtype float64 --energy-weight 1 --forces-weight 1 --stress-weight 0
```

Choose `--batch-max-edges` for your structures and available memory; the JAX
loader uses it to pack fixed-shape batches. Model bundles contain `config.json`
and `params.msgpack`. Each best-validation checkpoint is also a portable bundle.
Use a trained checkpoint directory as `--model` for JAX prediction/evaluation.
To resume optimizer and model state, keep the original model bundle and add
`--load-checkpoint PATH_TO_CHECKPOINT` to the training command.

Float64 parameters require `jax_enable_x64`; Equitrain enables it when
`--dtype float64` is selected. The GRACE graph adapter preserves geometry dtype,
and the graph loader carries its dtype into worker processes. A missing or zero
cell volume on a real structure raises in eager stress evaluation; compiled
evaluation produces NaN for invalid real stress. Zero-volume padded cells
produce zero stress and never enter the loss as real structures.

A local native Torch model can also supply identical JAX weights:

```python
import jax
from equitrain.models.grace import JaxGrace1L
from equitrain.models.grace.jax_model import save_bundle

jax.config.update('jax_enable_x64', True)
jax_native = JaxGrace1L(torch_native.config)
variables = jax_native.variables_from_torch(torch_native)
save_bundle('grace-jax-local', jax_native.config, variables=variables)
```

Here `torch_native` is a local `Grace1L` instance, or the `.model` of a
`GraceWrapper`. Conversion uses model parameters already in memory and performs
no downloads. JAX computation and training do not import Torch.

## Validation

The mandatory tests use small local models:

```bash
python -m pytest tests/test_grace_torch.py tests/test_train_grace.py \
  tests/test_grace_jax.py tests/test_train_grace_jax.py -q
```

They cover rotation/reflection, translation, atom permutation, independent
batching, zero net force, coordinate-axis derivatives, finite-difference forces
and periodic stress, cutoff smoothness, isolated atoms, derivative-only training,
configuration validation, checkpoint/export, and shared prediction integration.
The runtime tests verify backend isolation. JAX checks compare all supported
angular/correlation orders to identical local Torch weights, plus JIT energies,
forces, stress, mixed parameter gradients, symmetry, finite differences, padded
graphs, and training/checkpoint/resume/prediction through the real backend.

An optional reference suite builds fresh TensorFlow models and copies identical
random weights. It compares intermediate tensors, atomic energies, bond and
strain gradients, and parameter gradients of energy/derivative losses. It needs
the local upstream checkout and its dependencies, but no model downloads:

```bash
EQUITRAIN_GRACE_REFERENCE=grace-tensorpotential \
  python -m pytest tests/test_grace_reference.py -q
```

Before production use, run convergence checks on representative data, GPU
float32/float64 parity and memory/throughput measurements, multi-process training
and resume, and molecular dynamics energy-drift checks. These numerical tests
do not establish the accuracy of a model trained on a particular dataset.

## Angular table provenance and license

The fixed angular tables come from the upstream coupling generator with its
legacy history ordering, normalization, and exclusion list. They are independent
of species, channel widths, and trained weights. Regenerate them with:

```bash
python resources/models/grace/generate_couplings.py \
  grace-tensorpotential equitrain/models/grace/couplings.json
```

Regeneration needs NumPy, pandas, and SymPy; training does not need pandas or
SymPy for table generation. The native model adaptation and generated tables
retain the upstream **Academic Software Licence (ASL)**, including its academic
noncommercial restriction. Their license is included in
`equitrain/models/grace/LICENSE.md`; these files are not offered under Equitrain's
MIT license. Review distribution of this combined package against that license
before publishing it. Reference authors: Anton Bochkarev and Yury Lysogorskiy.
