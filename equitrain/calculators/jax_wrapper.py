"""Generic JAX-wrapper predictor and ASE calculator adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


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
    raise RuntimeError("`model_wrapper` is required (for example: mace, ani).")


def _normalize_jax_dtype(default_dtype: Any) -> str:
    if default_dtype is None:
        return "float32"
    value = str(default_dtype).strip().lower()
    aliases = {
        "float16": "float16",
        "half": "float16",
        "float32": "float32",
        "single": "float32",
        "float64": "float64",
        "double": "float64",
    }
    if value in aliases:
        return aliases[value]
    raise ValueError(
        f"Unsupported default_dtype '{default_dtype}'. "
        "Use one of: float16, float32, float64."
    )


def resolve_jax_device_or_fallback(
    requested_device: Any,
    *,
    logger=None,
    context: str = "runtime",
):
    try:
        import jax
    except Exception as exc:
        raise RuntimeError(
            "JAX is required for JAX calculators but could not be imported."
        ) from exc

    device = str(requested_device).strip() if requested_device is not None else "cpu"
    if not device:
        device = "cpu"
    dev_lower = device.lower()

    def _log(message: str) -> None:
        if callable(logger):
            logger(message)

    all_devices = list(jax.devices())
    if not all_devices:
        raise RuntimeError("No JAX devices are available.")

    def _devices_for_platform(platform: str) -> list[Any]:
        platform = platform.lower()
        if platform == "gpu":
            return [
                d
                for d in all_devices
                if str(getattr(d, "platform", "")).lower() in {"gpu", "cuda", "rocm"}
            ]
        return [
            d
            for d in all_devices
            if str(getattr(d, "platform", "")).lower() == platform
        ]

    if dev_lower == "auto":
        gpu_devices = _devices_for_platform("gpu")
        if gpu_devices:
            return gpu_devices[0]
        cpu_devices = _devices_for_platform("cpu")
        if cpu_devices:
            return cpu_devices[0]
        return all_devices[0]

    target_platform = None
    target_index = 0
    if dev_lower.startswith("cuda") or dev_lower.startswith("gpu"):
        target_platform = "gpu"
    elif dev_lower.startswith("cpu"):
        target_platform = "cpu"
    elif dev_lower.startswith("tpu"):
        target_platform = "tpu"

    if ":" in dev_lower:
        try:
            target_index = max(0, int(dev_lower.split(":", 1)[1]))
        except Exception:
            target_index = 0

    if target_platform is None:
        _log(
            f"Unrecognized JAX device '{device}' in {context}. "
            f"Using first available device '{all_devices[0]}'."
        )
        return all_devices[0]

    candidates = _devices_for_platform(target_platform)
    if not candidates:
        if target_platform == "gpu":
            _log(
                f"Requested GPU device '{device}' in {context}, but no JAX GPU device "
                "is available. Falling back to CPU."
            )
            cpu_devices = _devices_for_platform("cpu")
            if cpu_devices:
                return cpu_devices[0]
        _log(
            f"Requested JAX device '{device}' in {context}, but no matching platform "
            "device is available. Falling back to first available device."
        )
        return all_devices[0]

    if target_index >= len(candidates):
        _log(
            f"Requested JAX device '{device}' in {context}, but only "
            f"{len(candidates)} matching device(s) are visible. "
            "Using index 0 instead."
        )
        target_index = 0
    return candidates[target_index]


def _coerce_model_bundle(
    *,
    model: Any,
    model_wrapper: str,
    default_dtype: Any,
):
    try:
        from equitrain.backends.jax_utils import ModelBundle, load_model_bundle
    except Exception as exc:
        raise RuntimeError(
            "Missing JAX backend dependencies for calculator setup."
        ) from exc

    if isinstance(model, ModelBundle):
        return model

    has_bundle_fields = all(
        hasattr(model, attr) for attr in ("config", "params", "module")
    )
    if has_bundle_fields:
        return ModelBundle(
            config=dict(getattr(model, "config")),
            params=getattr(model, "params"),
            module=getattr(model, "module"),
        )

    model_path = str(model).strip()
    if not model_path:
        raise RuntimeError("Model path is empty.")
    path = Path(model_path).expanduser()
    if not path.exists():
        raise RuntimeError(
            "JAX calculator requires a JAX model bundle path or a loaded ModelBundle. "
            f"Got '{model_path}'."
        )
    dtype = _normalize_jax_dtype(default_dtype)
    return load_model_bundle(str(path), dtype=dtype, wrapper=model_wrapper)


class JaxWrapperPredictor:
    """Predict energies/forces for ASE atoms using equitrain JAX wrappers."""

    def __init__(
        self,
        *,
        model: Any,
        model_wrapper: str,
        device: str = "cpu",
        default_dtype: str | None = None,
        batch_size: int = 32,
        require_forces: bool = False,
        logger=None,
    ) -> None:
        self.logger = logger

        try:
            import jax
            import jax.numpy as jnp
            import jraph
            from equitrain.backends.jax_wrappers import create_wrapper
            from equitrain.data.backend_jax import AtomsToGraphs, make_apply_fn
        except Exception as exc:
            raise RuntimeError(
                "Missing JAX/jraph/equitrain dependencies for JAX calculator inference."
            ) from exc

        resolved_wrapper = _resolve_wrapper_name(model_wrapper=model_wrapper)
        bundle = _coerce_model_bundle(
            model=model,
            model_wrapper=resolved_wrapper,
            default_dtype=default_dtype,
        )
        wrapper = create_wrapper(
            resolved_wrapper,
            module=bundle.module,
            config=bundle.config,
            compute_force=bool(require_forces),
            compute_stress=False,
        )

        atomic_numbers = getattr(wrapper, "atomic_numbers", None)
        r_max = getattr(wrapper, "r_max", None)
        if atomic_numbers is None:
            raise RuntimeError(
                f"JAX wrapper '{resolved_wrapper}' does not expose `atomic_numbers`."
            )
        if r_max is None or float(r_max) <= 0.0:
            raise RuntimeError(
                f"JAX wrapper '{resolved_wrapper}' does not expose a valid `r_max`."
            )

        self.jax = jax
        self.jnp = jnp
        self.jraph = jraph
        self.params = bundle.params
        self.model_wrapper = resolved_wrapper
        self.batch_size = max(1, int(batch_size))
        self.require_forces = bool(require_forces)
        self.device = resolve_jax_device_or_fallback(
            device,
            logger=logger,
            context="equitrain-jax-calculator",
        )
        self.atoms_to_graphs = AtomsToGraphs(atomic_numbers, float(r_max))
        base_apply = make_apply_fn(wrapper, num_species=len(atomic_numbers))
        self._predict_fn = jax.jit(base_apply)

    def predict(
        self, atoms_list: list[Any], *, require_forces: bool
    ) -> tuple[list[float], list[np.ndarray] | None]:
        if require_forces and not self.require_forces:
            raise RuntimeError(
                "Predictor was initialized without forces support. "
                "Create it with require_forces=True."
            )
        if not atoms_list:
            return [], ([] if require_forces else None)

        energies: list[float] = []
        force_list: list[np.ndarray] | None = [] if require_forces else None

        for start in range(0, len(atoms_list), self.batch_size):
            atoms_chunk = atoms_list[start : start + self.batch_size]
            graphs = [
                self.atoms_to_graphs.convert(atoms.copy()) for atoms in atoms_chunk
            ]
            batch_graph = self.jraph.batch(graphs)
            host_graph = self.jax.tree_util.tree_map(
                lambda x: self.jnp.asarray(x),
                batch_graph,
                is_leaf=lambda x: x is None,
            )
            device_graph = self.jax.device_put(host_graph, self.device)
            outputs = self._predict_fn(self.params, device_graph)
            outputs = self.jax.device_get(outputs)

            energy_tensor = outputs.get("energy")
            if energy_tensor is None:
                raise RuntimeError("Model output did not contain 'energy'.")
            chunk_energies = np.asarray(energy_tensor).reshape(-1)
            if chunk_energies.shape[0] < len(atoms_chunk):
                raise RuntimeError(
                    f"Model returned only {chunk_energies.shape[0]} energies "
                    f"for {len(atoms_chunk)} graphs."
                )
            energies.extend([float(x) for x in chunk_energies[: len(atoms_chunk)]])

            if require_forces:
                force_tensor = outputs.get("forces")
                if force_tensor is None:
                    raise RuntimeError(
                        "Model output did not contain 'forces' required for relaxation."
                    )
                force_arr = np.asarray(force_tensor, dtype=float)
                n_node = np.asarray(batch_graph.n_node, dtype=int)
                total_nodes = int(n_node.sum())
                if force_arr.shape[0] < total_nodes:
                    raise RuntimeError(
                        f"Model returned {force_arr.shape[0]} force rows but "
                        f"{total_nodes} nodes were expected."
                    )
                if total_nodes == 0:
                    chunk_forces = [np.zeros((0, 3), dtype=float)] * len(atoms_chunk)
                else:
                    splits = np.cumsum(n_node)[:-1]
                    chunk_forces = np.split(force_arr[:total_nodes], splits, axis=0)
                for f in chunk_forces:
                    force_list.append(np.asarray(f, dtype=float))

        if (
            require_forces
            and force_list is not None
            and len(force_list) != len(energies)
        ):
            raise RuntimeError("Forces/energies count mismatch during prediction.")
        return energies, force_list


def build_jax_ase_calculator(
    *,
    model: Any,
    model_wrapper: str,
    device: str = "cpu",
    default_dtype: str | None = None,
    batch_size: int = 32,
    logger=None,
):
    """Build an ASE calculator backed by ``JaxWrapperPredictor``."""
    from ase.calculators.calculator import Calculator, all_changes

    predictor = JaxWrapperPredictor(
        model=model,
        model_wrapper=model_wrapper,
        device=device,
        default_dtype=default_dtype,
        batch_size=batch_size,
        require_forces=True,
        logger=logger,
    )

    class _EquitrainJaxASECalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self,
            atoms=None,
            properties=("energy", "forces"),
            system_changes=all_changes,
        ):
            super().calculate(atoms, properties, system_changes)
            energies, forces = predictor.predict([atoms], require_forces=True)
            self.results["energy"] = float(energies[0])
            self.results["forces"] = np.asarray(forces[0], dtype=float)

    return _EquitrainJaxASECalculator()


__all__ = [
    "JaxWrapperPredictor",
    "build_jax_ase_calculator",
    "resolve_jax_device_or_fallback",
]
