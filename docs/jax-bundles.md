# JAX Bundles

JAX workflows load model directories rather than Torch checkpoint files. A JAX
bundle contains:

```text
path/to/jax_bundle/
  config.json
  params.msgpack
```

Pass the bundle directory as `--model` and set `--backend jax`.

```bash
equitrain \
    --backend jax \
    --model path/to/jax_bundle \
    --model-wrapper mace \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --batch-max-edges 200000 \
    --output-dir runs/jax
```

## Common Config Keys

`config.json` should provide:

- `wrapper_name` or `model_wrapper`: wrapper name such as `mace`, `ani`, or
  `m3gnet`. An explicit `--model-wrapper` overrides this.
- `atomic_numbers`: atomic numbers covered by the model.
- `r_max`: graph cutoff radius. For M3GNet, `cutoff` is also accepted.
- `atomic_energies`: optional atomic energy offsets.

Custom JAX ANI and M3GNet bundles must also provide one of:

- `module_factory`
- `module_builder`
- `module_class`

The value must be an import string such as `my_package.my_model:create_model`.
`model_kwargs` is passed as keyword arguments to the factory/class.

## MACE-JAX

MACE-JAX bundles are built through the MACE-JAX conversion tools. This
repository includes a helper for supported foundation models:

```bash
python resources/models/mace-jax/convert_foundation_to_jax.py \
    --source mp \
    --model small \
    --output-dir resources/models/mace-jax/mp-small-jax
```

The generated directory can be passed directly to Equitrain's JAX backend.
MACE-JAX bundles can also be produced by compatible fine-tuning/checkpoint
utilities that write the same `config.json` and `params.msgpack` pair.

## JAX ANI Bundle Contract

JAX ANI uses a JAX-native ANI-like module. It does not load TorchANI checkpoints
directly.

Minimal `config.json`:

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

The wrapped module must expose an `apply` method. The wrapper can call modules
that accept either:

- a mapping with `species`, `coordinates`, `atom_mask`, and `counts`; or
- positional `(species, coordinates)` inputs.

The output must provide energy and may provide forces and stress. Accepted
return styles include:

- a mapping with `energy`, optional `forces`, and optional `stress`;
- an object with an `energies` attribute and optional `forces`/`stress`;
- a tuple/list whose second item is the energy tensor.

If the module returns only energy and `--forces-weight` is positive, the wrapper
computes forces with `jax.grad`.

Start by testing JAX ANI with `--forces-weight 0.0` for an energy-only smoke
test. After the bundle loads and energy training works, enable force training to
exercise the gradient force path.

Example JAX ANI training command:

```bash
equitrain -v \
    --backend jax \
    --model path/to/jax_ani_bundle \
    --model-wrapper ani \
    --train-file data/train.h5 \
    --valid-file data/valid.h5 \
    --output-dir runs/jax-ani \
    --energy-weight 1.0 \
    --forces-weight 1.0 \
    --stress-weight 0.0 \
    --batch-max-edges 10000 \
    --epochs 5
```

## JAX M3GNet Bundle Contract

JAX M3GNet uses a JAX-native graph module. It does not load MatGL Torch
checkpoints directly.

Minimal `config.json`:

```json
{
  "wrapper_name": "m3gnet",
  "atomic_numbers": [1, 6, 7, 8],
  "element_types": ["H", "C", "N", "O"],
  "r_max": 5.0,
  "module_factory": "my_package.my_m3gnet:create_model",
  "model_kwargs": {}
}
```

The module receives a flat graph dictionary. Equitrain preserves its original
keys and adds MatGL-like aliases:

- `positions` / `pos`
- `node_attrs_index` / `node_type` / `species`
- `edge_index`, `senders`, `receivers`
- `shifts` / `pbc_offshift`
- `unit_shifts` / `pbc_offset`
- `batch`, `edge_batch`, `ptr`
- `cell`
- `node_mask`, `graph_mask`

The output must provide energy and may provide forces and stress. Accepted
return styles include:

- a mapping with `energy`, optional `forces`, and optional `stress`;
- an object with `energy` or `energies` and optional `forces`/`stress`;
- a tuple/list containing energy, optional forces, and optional stress.

If the module returns only energy and `--forces-weight` is positive, the wrapper
computes forces with `jax.grad`. If stress is requested and not returned by the
module, the wrapper differentiates a strain-displacement route and requires
`cell` in the input data.

For custom modules whose parameter tree cannot be inferred through NNX splitting,
the factory may return `(module, params_template)`.

## Multi-Device Notes

- On single-device machines, the JAX backend uses the normal single-device path.
- With more than one global JAX device, training/evaluation automatically uses
  multi-device `shard_map` execution.
- After `jax.distributed.initialize()`, the device mesh spans `jax.devices()`,
  so gradient and metric collectives synchronize across nodes as well as
  devices on one node.
- Each process provides local micro-batches for `jax.local_device_count()`
  devices; Equitrain converts them into globally sharded arrays.
- For multi-node jobs, launch one Equitrain process per JAX process with
  `--distributed --launcher none`, `--process-count <global-processes>`,
  `--process-index <rank>`, and `--coordinator-address <host:port>`. A process
  may own one or more local devices; the local launcher is intended for
  single-node multi-GPU runs.
