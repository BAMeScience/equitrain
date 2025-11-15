"""
ORB wrapper with optional dependency handling.
"""

from __future__ import annotations

import warnings

import torch

from equitrain.data.atomic import AtomicNumberTable

from .base import AbstractWrapper

try:
    from ase import Atoms
except Exception as exc:  # pragma: no cover - ase is a core dependency for orb tests
    raise ImportError('ase is required for OrbWrapper.') from exc


class OrbWrapper(AbstractWrapper):
    """
    Wrapper for ORB (Orbital Materials) models.

    orb-models>=0.5 expects the input as an `AtomGraphs` object.  The torch backend
    operates on Torch Geometric `Data`/`Batch` instances, so this wrapper performs the
    conversion and exposes a uniform dictionary with `energy`, `forces`, and `stress`
    tensors to the shared training pipeline.
    """

    def __init__(self, args, model=None, model_variant='direct', enable_zbl=False):
        super().__init__(model)

        self.compute_force = getattr(args, 'forces_weight', 0.0) > 0.0
        self.compute_stress = getattr(args, 'stress_weight', 0.0) > 0.0
        self.model_variant = model_variant
        self.enable_zbl = enable_zbl

        if model is None:
            self.model = self._create_default_model()

        self.system_config = getattr(self.model, 'system_config', None)
        if self.system_config is None:
            from orb_models.forcefield.atomic_system import SystemConfig

            self.system_config = SystemConfig(radius=6.0, max_num_neighbors=20)

        self.model_dtype = next(self.model.parameters()).dtype
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad_(True)

    def _create_default_model(self):
        try:
            from orb_models.forcefield import pretrained
        except ImportError as exc:
            raise ImportError(
                'ORB models are required. Install with: pip install "orb-models>=0.5"'
            ) from exc

        if self.model_variant == 'direct':
            candidate_names = [
                'orb_v3_direct_20_mpa',
                'orb_v3_direct_inf_mpa',
                'orb_v2',
                'orb_v1',
            ]
        else:
            candidate_names = [
                'orb_v3_conservative_20_mpa',
                'orb_v3_conservative_inf_mpa',
            ]

        for name in candidate_names:
            factory = getattr(pretrained, name, None)
            if callable(factory):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            'ignore',
                            message='Setting global torch default dtype to torch.float32.',
                            category=UserWarning,
                        )
                        return factory()
                except Exception:
                    continue

        raise RuntimeError(
            'Could not instantiate an ORB model. Please ensure orb-models>=0.5 is '
            'installed together with the pretrained weights cache.'
        )

    def _ensure_batch_from_args(self, *args):
        if not args:
            raise TypeError('OrbWrapper.forward() missing inputs')

        if len(args) == 1:
            return args[0]

        atomic_numbers = args[0]
        positions = args[1].to(self.model_dtype)
        lattice = args[2] if len(args) > 2 else None
        batch = args[3] if len(args) > 3 else None

        try:
            from torch_geometric.data import Batch, Data
        except ImportError as exc:
            raise ImportError(
                'torch_geometric is required to batch OrbWrapper inputs.'
            ) from exc

        if batch is None:
            graph = Data(
                atomic_numbers=atomic_numbers,
                positions=positions,
                cell=lattice.to(self.model_dtype).unsqueeze(0)
                if lattice is not None
                else None,
                pbc=torch.zeros(3, dtype=torch.bool, device=positions.device),
            )
            graph.idx = 0
            return Batch.from_data_list([graph])

        # Multi-structure input
        graphs = []
        for idx in torch.unique(batch, sorted=True).tolist():
            mask = batch == idx
            graph = Data(
                atomic_numbers=atomic_numbers[mask],
                positions=positions[mask],
                cell=lattice[idx].unsqueeze(0).to(self.model_dtype)
                if lattice is not None
                else None,
                pbc=torch.zeros(3, dtype=torch.bool, device=positions.device),
            )
            graph.idx = idx
            graphs.append(graph)
        return Batch.from_data_list(graphs)

    def _cast_batch_to_model_dtype(self, batch):
        attributes = ['positions', 'cell', 'velocities', 'atomic_numbers']
        for attr in attributes:
            value = getattr(batch, attr, None)
            if value is not None and torch.is_floating_point(value):
                setattr(batch, attr, value.to(self.model_dtype))

        for attr in ('y', 'force', 'stress'):
            value = getattr(batch, attr, None)
            if value is not None and torch.is_tensor(value):
                setattr(batch, attr, value.to(self.model_dtype))

        return batch

    def _data_to_atom_graphs(self, data):
        try:
            from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError(
                'orb-models>=0.5 is required to run the OrbWrapper.'
            ) from exc

        device = data.positions.device
        dtype = self.model_dtype

        numbers = data.atomic_numbers.detach().cpu().numpy()
        positions = data.positions.detach().cpu().numpy()
        cell_tensor = getattr(data, 'cell', None)
        cell = cell_tensor.detach().cpu().numpy() if cell_tensor is not None else None

        pbc_tensor = getattr(data, 'pbc', None)
        if pbc_tensor is not None:
            pbc = pbc_tensor.detach().cpu().numpy()
        else:
            pbc = False

        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)

        tags_tensor = getattr(data, 'tags', None)
        if tags_tensor is not None:
            atoms.set_tags(tags_tensor.detach().cpu().numpy())

        node_attrs = data.get('node_attrs', None) if hasattr(data, 'get') else None
        if node_attrs is not None:
            atoms.info.setdefault('node_features', {})
            atoms.info['node_features']['atomic_numbers_embedding'] = (
                node_attrs.detach().cpu()
            )

        graph = ase_atoms_to_atom_graphs(
            atoms,
            self.system_config,
            wrap=False,
            device=device,
            output_dtype=dtype,
        )

        if self.enable_zbl and hasattr(self.model, 'zbl_model'):
            graph = graph.with_zbl()

        return graph

    def _as_atom_graphs(self, data):
        try:
            from torch_geometric.data import Batch
        except ImportError as exc:
            raise ImportError(
                'torch_geometric is required to batch OrbWrapper inputs.'
            ) from exc

        if isinstance(data, Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]

        atom_graphs = [self._data_to_atom_graphs(item) for item in data_list]
        if len(atom_graphs) == 1:
            return atom_graphs[0]

        from orb_models.forcefield.atomic_system import batch_graphs

        return batch_graphs(atom_graphs)

    def forward(self, *args):
        """Run the wrapped ORB model and return energy/forces/stress tensors."""
        data = self._ensure_batch_from_args(*args)
        data = self._cast_batch_to_model_dtype(data)
        atom_graphs = self._as_atom_graphs(data)

        dtype = atom_graphs.node_features['positions'].dtype

        autocast_enabled = torch.cuda.is_available()
        with torch.amp.autocast('cuda', enabled=autocast_enabled):
            outputs = self.model(atom_graphs)

        energy = outputs.get('energy')
        if energy is None:
            raise KeyError('ORB model output is missing an `energy` tensor.')
        energy = energy.view(-1).to(dtype)

        forces = None
        if self.compute_force:
            force_key = 'grad_forces' if 'grad_forces' in outputs else 'forces'
            forces = outputs.get(force_key)
            if forces is None:
                raise KeyError(
                    f'ORB model output is missing `{force_key}` predictions.'
                )
            forces = forces.to(dtype)

        stress = None
        if self.compute_stress:
            stress_key = 'grad_stress' if 'grad_stress' in outputs else 'stress'
            stress_voigt = outputs.get(stress_key)
            if stress_voigt is None:
                raise KeyError(
                    f'ORB model output is missing `{stress_key}` predictions.'
                )
            if stress_voigt.dim() == 1:
                stress_voigt = stress_voigt.unsqueeze(0)

            from orb_models.forcefield.forcefield_utils import (
                torch_voigt_6_to_full_3x3_stress,
            )

            stress = torch_voigt_6_to_full_3x3_stress(stress_voigt.to(dtype))

        return {'energy': energy, 'forces': forces, 'stress': stress}

    @property
    def atomic_numbers(self):
        if hasattr(self.model, 'atomic_numbers'):
            return AtomicNumberTable(self.model.atomic_numbers.cpu().tolist())
        return AtomicNumberTable(list(range(1, 85)))

    @property
    def atomic_energies(self):
        if hasattr(self.model, 'atomic_energies'):
            return self.model.atomic_energies.cpu().tolist()
        return None

    @property
    def r_max(self):
        if self.system_config is not None:
            return float(self.system_config.radius)
        if hasattr(self.model, 'cutoff'):
            cutoff = self.model.cutoff
            return float(cutoff.item()) if torch.is_tensor(cutoff) else float(cutoff)
        return 6.0

    @r_max.setter
    def r_max(self, value):
        if self.system_config is not None:
            self.system_config.radius = float(value)
        if hasattr(self.model, 'cutoff'):
            cutoff = self.model.cutoff
            if torch.is_tensor(cutoff):
                self.model.cutoff.fill_(value)
            else:
                self.model.cutoff = value
        elif hasattr(self.model, 'r_max'):
            r_max_tensor = self.model.r_max
            if torch.is_tensor(r_max_tensor):
                r_max_tensor.fill_(value)
            else:
                self.model.r_max = value


__all__ = ['OrbWrapper']
