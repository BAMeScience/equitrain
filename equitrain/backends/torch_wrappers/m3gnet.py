"""
MatGL-backed M3GNet wrapper for the torch backend.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

warnings.filterwarnings('ignore', message='cuaev not installed', category=UserWarning)
warnings.filterwarnings(
    'ignore', message='pkg_resources is deprecated as an API', category=UserWarning
)
warnings.filterwarnings(
    'ignore', message="module 'sre_parse' is deprecated", category=DeprecationWarning
)
warnings.filterwarnings(
    'ignore',
    message="module 'sre_constants' is deprecated",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    'ignore',
    message='You are using `torch.load` with `weights_only=False`',
    category=FutureWarning,
)

import dgl
import torch
from ase.data import atomic_numbers as ASE_ATOMIC_NUMBERS

from equitrain.backends.torch_derivatives.force import compute_force
from equitrain.backends.torch_derivatives.stress import (
    compute_stress,
    get_displacement,
)
from equitrain.data.atomic import AtomicNumberTable

from .base import AbstractWrapper

try:  # pragma: no cover - optional dependency resolution
    from matgl.config import DEFAULT_ELEMENTS
    from matgl.models import M3GNet as MatGLM3GNet
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        'The MatGL-backed M3GNet wrapper requires the `matgl` package together with a functional `dgl` install.'
    ) from exc


class M3GNetWrapper(AbstractWrapper):
    """
    Thin wrapper around MatGL's M3GNet implementation.
    """

    def __init__(self, args, model=None, element_types: Iterable[str] | None = None):
        resolved_elements = self._resolve_element_types(model, element_types)

        if model is None:
            model = MatGLM3GNet(element_types=resolved_elements)

        super().__init__(model)  # type: ignore[arg-type]

        if hasattr(args, 'dtype') and getattr(args, 'dtype') != 'float32':
            args.dtype = 'float32'

        self.compute_force = bool(getattr(args, 'forces_weight', 0.0) > 0.0)
        self.compute_stress = bool(getattr(args, 'stress_weight', 0.0) > 0.0)

        self._element_types = resolved_elements
        self._atomic_number_lookup = self._build_atomic_number_lookup(
            self._element_types
        )
        atomic_numbers = list(self._atomic_number_lookup.keys())
        self._atomic_numbers_table = AtomicNumberTable(atomic_numbers)

        lookup_tensor = torch.full((max(atomic_numbers) + 1,), -1, dtype=torch.long)
        for z, idx in self._atomic_number_lookup.items():
            lookup_tensor[z] = idx
        self.register_buffer('_z_to_index', lookup_tensor, persistent=False)

        atomic_energies = getattr(self.model, 'atomic_energies', None)
        if atomic_energies is not None:
            atomic_energies = torch.as_tensor(
                atomic_energies, dtype=torch.get_default_dtype()
            )
        else:
            atomic_energies = torch.zeros(
                len(self._atomic_number_lookup), dtype=torch.get_default_dtype()
            )
        self.register_buffer('_atomic_energies', atomic_energies, persistent=False)

        self._cutoff = float(getattr(self.model, 'cutoff', 5.0))

    @staticmethod
    def _resolve_element_types(
        model=None, element_types: Iterable[str] | None = None
    ) -> tuple[str, ...]:
        if element_types is not None:
            return tuple(str(symbol).strip() for symbol in element_types)
        if model is not None and hasattr(model, 'element_types'):
            return tuple(getattr(model, 'element_types'))
        return tuple(DEFAULT_ELEMENTS)

    @staticmethod
    def _build_atomic_number_lookup(element_types: Iterable[str]) -> dict[int, int]:
        lookup: dict[int, int] = {}
        for idx, symbol in enumerate(element_types):
            if symbol not in ASE_ATOMIC_NUMBERS:
                raise ValueError(f'Unknown chemical symbol: {symbol}')
            lookup[int(ASE_ATOMIC_NUMBERS[symbol])] = idx
        return lookup

    def _map_atomic_numbers(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        if atomic_numbers.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=atomic_numbers.device)

        if atomic_numbers.max().item() >= self._z_to_index.shape[0]:
            raise ValueError(
                'Encountered atomic number outside the supported element set.'
            )

        mapped = self._z_to_index[atomic_numbers.long()]
        if torch.any(mapped < 0):
            missing = torch.unique(atomic_numbers[mapped < 0])
            raise ValueError(
                f'Atomic numbers {missing.tolist()} are not supported by this model.'
            )
        return mapped

    def forward(self, *args):
        if len(args) != 1:
            raise NotImplementedError(
                'M3GNetWrapper expects a single PyG batch argument.'
            )

        data = args[0]
        for attr in ('positions', 'atomic_numbers', 'batch', 'ptr', 'edge_index'):
            if not hasattr(data, attr):
                raise ValueError(
                    f'Input data is missing required attribute `{attr}` for M3GNetWrapper.'
                )

        num_graphs = int(data.ptr.shape[0] - 1)
        param = next(self.model.parameters(), None)
        target_dtype = param.dtype if param is not None else data.positions.dtype

        if hasattr(data, 'y') and data.y is not None:
            data.y = data.y.to(dtype=target_dtype)
        if hasattr(data, 'force') and data.force is not None:
            data.force = data.force.to(dtype=target_dtype)
        if hasattr(data, 'stress') and data.stress is not None:
            data.stress = data.stress.to(dtype=target_dtype)

        positions = data.positions.to(dtype=target_dtype)
        if self.compute_force or self.compute_stress:
            positions = positions.clone().detach().requires_grad_(True)

        batch = data.batch
        atomic_numbers = data.atomic_numbers
        species_indices = self._map_atomic_numbers(atomic_numbers)

        ptr = data.ptr
        edge_src = data.edge_index[0]
        edge_dst = data.edge_index[1]
        edge_batch = batch[edge_src]
        shifts = data.shifts.to(dtype=target_dtype)
        unit_shifts = data.unit_shifts.to(dtype=target_dtype)

        graphs: list[dgl.DGLGraph] = []
        for graph_idx in range(num_graphs):
            start = int(ptr[graph_idx].item())
            end = int(ptr[graph_idx + 1].item())
            node_slice = slice(start, end)
            node_count = end - start
            mask = edge_batch == graph_idx

            local_src = edge_src[mask] - start
            local_dst = edge_dst[mask] - start

            graph = dgl.graph(
                (local_src, local_dst),
                num_nodes=node_count,
                device=positions.device,
            )
            graph.ndata['node_type'] = species_indices[node_slice]
            graph.ndata['pos'] = positions[node_slice]
            graph.edata['pbc_offshift'] = shifts[mask]
            graph.edata['pbc_offset'] = unit_shifts[mask]
            graphs.append(graph)

        displacement = None
        if self.compute_stress:
            positions, displacement = get_displacement(positions, num_graphs, batch)
            cursor = 0
            for graph in graphs:
                count = graph.num_nodes()
                graph.ndata['pos'] = positions[cursor : cursor + count]
                cursor += count

        batched_graph = dgl.batch(graphs)
        try:  # pragma: no cover - optional dependency guard
            import matgl

            if getattr(matgl, 'float_th', None) != target_dtype:
                matgl.float_th = target_dtype  # type: ignore[attr-defined]
        except Exception:
            pass
        energy = self.model(batched_graph)

        y_pred: dict[str, torch.Tensor | None] = {
            'energy': energy,
            'forces': None,
            'stress': None,
        }

        if self.compute_force:
            forces = compute_force(
                energy,
                positions,
                training=self.training or self.compute_stress,
            )
            y_pred['forces'] = forces

        if self.compute_stress:
            cell = data.cell.to(dtype=target_dtype)
            stress = compute_stress(
                energy,
                displacement,
                cell,
                training=self.training,
            )
            y_pred['stress'] = stress

        return y_pred

    @property
    def atomic_numbers(self):
        return self._atomic_numbers_table

    @property
    def atomic_energies(self):
        return self._atomic_energies

    @property
    def r_max(self):
        return float(getattr(self.model, 'cutoff', self._cutoff))

    @r_max.setter
    def r_max(self, value):
        self._cutoff = float(value)
        setattr(self.model, 'cutoff', float(value))


__all__ = ['M3GNetWrapper']
