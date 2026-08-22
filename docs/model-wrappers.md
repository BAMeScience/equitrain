# Model Wrappers

Model wrappers adapt backend-specific model objects to Equitrain's common
training, evaluation, and prediction interfaces.

## Supported Wrappers

| Wrapper | Backends | Model Artifact |
| --- | --- | --- |
| `mace` | Torch, JAX | Torch MACE model file or JAX bundle. |
| `sevennet` | Torch | SevenNet model/checkpoint. |
| `orb` | Torch | ORB model object or checkpoint. |
| `ani` | Torch, JAX | TorchANI model/checkpoint or JAX-native ANI bundle. |
| `m3gnet` | Torch, JAX | MatGL-backed Torch model or JAX-native M3GNet bundle. |

## MACE

For MACE, use the companion
[`mace-model`](https://github.com/bamescience/mace-model) repository for model
definition, initialization, conversion, and foundation-model export. Use
Equitrain for preprocessing, training, fine-tuning, checkpointing, evaluation,
and prediction.

JAX MACE uses a bundle containing `config.json` and `params.msgpack`. The
resource helper at `resources/models/mace-jax/convert_foundation_to_jax.py`
converts supported Torch MACE foundation models into this format.

## ORB

Install the ORB extra:

```bash
pip install 'equitrain[orb]'
```

The ORB resource directory contains example code and a configuration
sketch:

- `resources/models/orb/README.md`
- `resources/models/orb/orb_config.yaml`

## ANI

Torch ANI uses TorchANI models/checkpoints directly and requires the `ani`
extra:

```bash
pip install 'equitrain[torch,ani]'
```

The helper at `resources/models/ani/ani-initial-model.py` exports one of the
available TorchANI pretrained model families to an Equitrain-compatible
checkpoint.

JAX ANI is separate: it uses a JAX-native ANI-like bundle and does not load
TorchANI checkpoints directly. See [JAX Bundles](jax-bundles.md).

## M3GNet

Torch M3GNet uses MatGL:

```bash
pip install 'equitrain[m3gnet]'
```

The resource directory contains example code and a configuration sketch:

- `resources/models/m3gnet/README.md`
- `resources/models/m3gnet/m3gnet-config.yaml`

JAX M3GNet uses a JAX-native bundle and does not load MatGL Torch checkpoints
directly. See [JAX Bundles](jax-bundles.md).
