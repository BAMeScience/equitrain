"""Generic torch-wrapper predictor and ASE calculator adapters."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def resolve_device_or_fallback(
    requested_device: Any, *, logger=None, context: str = 'runtime'
) -> str:
    device = str(requested_device).strip() if requested_device is not None else 'cpu'
    if not device:
        device = 'cpu'

    dev_lower = device.lower()
    try:
        import torch
    except Exception:
        if dev_lower.startswith('cuda'):
            if callable(logger):
                logger(
                    f"Requested CUDA device '{device}' in {context}, but PyTorch CUDA "
                    'backend is unavailable. Falling back to cpu.'
                )
            return 'cpu'
        if dev_lower == 'auto':
            return 'cpu'
        return device

    if dev_lower == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if not dev_lower.startswith('cuda'):
        return device

    if not torch.cuda.is_available():
        if callable(logger):
            logger(
                f"Requested CUDA device '{device}' in {context}, but "
                'torch.cuda.is_available() is False. Falling back to cpu.'
            )
        return 'cpu'

    if ':' in dev_lower:
        try:
            cuda_idx = int(dev_lower.split(':', 1)[1])
            n_cuda = int(torch.cuda.device_count())
            if cuda_idx < 0 or cuda_idx >= n_cuda:
                if callable(logger):
                    logger(
                        f"Requested CUDA device '{device}' in {context}, but only "
                        f'{n_cuda} CUDA device(s) are visible. Falling back to cpu.'
                    )
                return 'cpu'
        except Exception:
            pass

    return device


def _resolve_torch_dtype(default_dtype: Any, torch) -> Any:
    if default_dtype is None:
        return None
    value = str(default_dtype).strip().lower()
    mapper = {
        'float16': torch.float16,
        'half': torch.float16,
        'float32': torch.float32,
        'single': torch.float32,
        'float64': torch.float64,
        'double': torch.float64,
    }
    if value in mapper:
        return mapper[value]
    raise ValueError(
        f"Unsupported default_dtype '{default_dtype}'. "
        'Use one of: float16, float32, float64.'
    )


def _normalize_wrapper_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _resolve_wrapper_name(
    *,
    model_wrapper: Any,
) -> str:
    resolved = _normalize_wrapper_name(model_wrapper)
    if resolved:
        return resolved
    raise RuntimeError(
        '`model_wrapper` is required (for example: mace, ani, orb, sevennet, m3gnet).'
    )


def _build_wrapped_torch_model(
    *,
    model: Any,
    model_wrapper: str,
    device: str,
    default_dtype: str | None,
    require_forces: bool,
    logger=None,
) -> tuple[Any, str, str]:
    try:
        import torch
        from equitrain.backends.torch_model import get_model as get_torch_model
    except Exception as exc:
        raise RuntimeError(
            'Missing torch/equitrain torch backend dependencies for calculator setup.'
        ) from exc

    resolved_device = resolve_device_or_fallback(
        device, logger=logger, context='equitrain-calculator'
    )
    if isinstance(model, torch.nn.Module):
        source_model = model
    else:
        model_path = str(model).strip()
        if not model_path:
            raise RuntimeError('Model path is empty.')
        path = Path(model_path).expanduser()
        if not path.exists():
            raise RuntimeError(
                'Calculator requires a concrete model object or an existing model '
                f"path. Got '{model_path}'. Resolve foundation aliases outside the "
                'calculator and pass the loaded model object.'
            )
        if path.is_dir():
            raise RuntimeError(
                'Calculator expects a model file path (or torch.nn.Module), '
                f"but got directory: '{path}'. Resolve bundle directories to a "
                'loaded model object before calling calculators.'
            )
        source_model = str(path)

    resolved_wrapper = _resolve_wrapper_name(model_wrapper=model_wrapper)

    wrapper_args = SimpleNamespace(
        model=source_model,
        model_wrapper=resolved_wrapper,
        energy_weight=1.0,
        forces_weight=1.0 if require_forces else 0.0,
        stress_weight=0.0,
        r_max=None,
    )
    wrapped_model = get_torch_model(wrapper_args)

    target_dtype = _resolve_torch_dtype(default_dtype, torch)
    if target_dtype is not None:
        wrapped_model = wrapped_model.to(dtype=target_dtype)
    wrapped_model = wrapped_model.to(resolved_device)
    wrapped_model.eval()

    atomic_numbers = getattr(wrapped_model, 'atomic_numbers', None)
    r_max = getattr(wrapped_model, 'r_max', None)
    if atomic_numbers is None:
        raise RuntimeError(
            f"Wrapped model for wrapper '{resolved_wrapper}' does not expose "
            '`atomic_numbers`, required for graph conversion.'
        )
    if r_max is None:
        raise RuntimeError(
            f"Wrapped model for wrapper '{resolved_wrapper}' does not expose "
            '`r_max`, required for graph construction cutoff.'
        )

    return wrapped_model, str(resolved_device), resolved_wrapper


class TorchWrapperPredictor:
    """Predict energies/forces for ASE atoms using equitrain torch wrappers."""

    def __init__(
        self,
        *,
        model: Any,
        model_wrapper: str,
        device: str = 'cpu',
        default_dtype: str | None = None,
        batch_size: int = 32,
        require_forces: bool = False,
        logger=None,
    ) -> None:
        self.logger = logger

        try:
            import torch
            from torch_geometric.data import Batch
            from equitrain.data.backend_torch.atoms_to_graphs import AtomsToGraphs
        except Exception as exc:
            raise RuntimeError(
                'Missing torch/torch_geometric/equitrain dependencies for '
                'torch-wrapper inference.'
            ) from exc

        self.torch = torch
        self.Batch = Batch
        self.batch_size = max(1, int(batch_size))
        self.require_forces = bool(require_forces)

        self.model, resolved_device, self.model_wrapper = _build_wrapped_torch_model(
            model=model,
            model_wrapper=model_wrapper,
            device=device,
            default_dtype=default_dtype,
            require_forces=self.require_forces,
            logger=logger,
        )
        self.device = torch.device(str(resolved_device))
        self.r_max = float(self.model.r_max)
        self.z_table = self.model.atomic_numbers
        self.atoms_to_graphs = AtomsToGraphs(
            self.z_table,
            radius=self.r_max,
            r_distances=True,
            r_edges=True,
            r_pbc=True,
        )

    def _predict_batch(
        self,
        batch,
        *,
        require_forces: bool,
    ) -> tuple[list[float], list[np.ndarray] | None]:
        pred = self.model(batch)
        energy_tensor = pred.get('energy')
        if energy_tensor is None:
            raise RuntimeError("Model output did not contain 'energy'.")
        energies = [float(x) for x in energy_tensor.reshape(-1).detach().cpu().tolist()]

        if not require_forces:
            return energies, None

        forces_tensor = pred.get('forces')
        if forces_tensor is None:
            raise RuntimeError(
                "Model output did not contain 'forces' required for relaxation."
            )
        forces_cpu = forces_tensor.detach().cpu()
        ptr = batch.ptr.detach().cpu().tolist()
        force_list: list[np.ndarray] = []
        for i in range(len(ptr) - 1):
            force_list.append(np.asarray(forces_cpu[ptr[i] : ptr[i + 1]], dtype=float))
        return energies, force_list

    def predict(
        self, atoms_list: list[Any], *, require_forces: bool
    ) -> tuple[list[float], list[np.ndarray] | None]:
        if require_forces and not self.require_forces:
            raise RuntimeError(
                'Predictor was initialized without forces support. '
                'Create it with require_forces=True.'
            )

        if not atoms_list:
            return [], ([] if require_forces else None)

        energies: list[float] = []
        force_list: list[np.ndarray] | None = [] if require_forces else None
        grad_context = nullcontext() if require_forces else self.torch.no_grad()
        with grad_context:
            if len(atoms_list) == 1:
                batch = self.Batch.from_data_list(
                    [self.atoms_to_graphs.convert(atoms_list[0])]
                ).to(self.device)
                batch_energies, batch_forces = self._predict_batch(
                    batch, require_forces=require_forces
                )
                energies.extend(batch_energies)
                if (
                    require_forces
                    and force_list is not None
                    and batch_forces is not None
                ):
                    force_list.extend(batch_forces)
            else:
                for start in range(0, len(atoms_list), self.batch_size):
                    end = min(start + self.batch_size, len(atoms_list))
                    chunk_graphs = [
                        self.atoms_to_graphs.convert(atoms)
                        for atoms in atoms_list[start:end]
                    ]
                    batch = self.Batch.from_data_list(chunk_graphs).to(self.device)
                    batch_energies, batch_forces = self._predict_batch(
                        batch, require_forces=require_forces
                    )
                    energies.extend(batch_energies)
                    if (
                        require_forces
                        and force_list is not None
                        and batch_forces is not None
                    ):
                        force_list.extend(batch_forces)

        if (
            require_forces
            and force_list is not None
            and len(force_list) != len(energies)
        ):
            raise RuntimeError('Forces/energies count mismatch during prediction.')
        return energies, force_list


def build_ase_calculator(
    *,
    model: Any,
    model_wrapper: str,
    device: str = 'cpu',
    default_dtype: str | None = None,
    batch_size: int = 32,
    logger=None,
):
    """Build an ASE calculator backed by ``TorchWrapperPredictor``."""
    from ase.calculators.calculator import Calculator, all_changes

    predictor = TorchWrapperPredictor(
        model=model,
        model_wrapper=model_wrapper,
        device=device,
        default_dtype=default_dtype,
        batch_size=batch_size,
        require_forces=True,
        logger=logger,
    )

    class _EquitrainASECalculator(Calculator):
        implemented_properties = ['energy', 'forces']

        def calculate(
            self,
            atoms=None,
            properties=('energy', 'forces'),
            system_changes=all_changes,
        ):
            super().calculate(atoms, properties, system_changes)
            energies, forces = predictor.predict([atoms], require_forces=True)
            self.results['energy'] = float(energies[0])
            self.results['forces'] = np.asarray(forces[0], dtype=float)

    return _EquitrainASECalculator()


__all__ = [
    'TorchWrapperPredictor',
    'build_ase_calculator',
    'resolve_device_or_fallback',
]
