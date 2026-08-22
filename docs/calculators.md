# Calculators

Equitrain exposes calculator helpers for structure-level inference and ASE
relaxation.

Torch:

- `equitrain.calculators.TorchWrapperPredictor`
- `equitrain.calculators.build_ase_calculator`

JAX:

- `equitrain.calculators.JaxWrapperPredictor`
- `equitrain.calculators.build_jax_ase_calculator`

The same helpers are also available from the top-level package:

```python
from equitrain import build_ase_calculator, get_torch_wrapper_predictor
```

## Behavior

- `model_wrapper` must be explicit.
- Torch `model` can be a loaded `torch.nn.Module` or an existing model file
  path.
- JAX `model` can be a loaded `ModelBundle` or a bundle path.
- Foundation-model aliases should be resolved before creating a calculator.
- Calculators return energy and forces; stress is not returned.
- If a requested CUDA/GPU device is unavailable, the API falls back to CPU or
  the first available JAX device.

Supported wrappers:

- Torch calculator: `mace`, `ani`, `orb`, `sevennet`, `m3gnet`
- JAX calculator: wrappers available in `equitrain.backends.jax_wrappers`,
  currently `mace`, `ani`, and `m3gnet`

## ASE Geometry Optimization

```python
from ase.build import molecule
from ase.optimize import FIRE
from equitrain.calculators import build_ase_calculator

atoms = molecule('H2O')
atoms.calc = build_ase_calculator(
    model='path/to/model.pt',
    model_wrapper='mace',
    device='cuda:0',
    batch_size=8,
)

opt = FIRE(atoms, logfile=None)
opt.run(fmax=0.05, steps=200)
print(atoms.get_potential_energy())
```

## Batched Torch Prediction

```python
from ase.build import molecule
from equitrain.calculators import TorchWrapperPredictor

predictor = TorchWrapperPredictor(
    model='path/to/model.pt',
    model_wrapper='mace',
    device='cuda:0',
    default_dtype='float32',
    batch_size=16,
    require_forces=True,
)

atoms = [molecule('H2O'), molecule('CH4')]
energies, forces = predictor.predict(atoms, require_forces=True)
```

## JAX Calculator

```python
from ase.build import molecule
from equitrain.calculators import build_jax_ase_calculator

atoms = molecule('H2O')
atoms.calc = build_jax_ase_calculator(
    model='path/to/jax_bundle',
    model_wrapper='mace',
    device='cpu',
)
print(atoms.get_potential_energy())
```

For batched JAX prediction, use `JaxWrapperPredictor` with a JAX bundle path or
a loaded `ModelBundle`.
