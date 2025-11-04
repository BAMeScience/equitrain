"""
Torch-specific model wrappers.

These modules adapt raw Torch models to the interface expected by the shared
training/evaluation logic. They are imported by the torch backend and re-exported
from the legacy ``equitrain.model_wrappers`` module for compatibility.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import torch

from equitrain.data.atomic import AtomicNumberTable

try:
    from mace.modules.blocks import RadialEmbeddingBlock
    from mace.modules.radial import (
        AgnesiTransform,
        BesselBasis,
        ChebychevBasis,
        GaussianBasis,
        SoftTransform,
        ZBLBasis,
    )

    _HAS_MACE = True
except ImportError:
    _HAS_MACE = False


class AbstractWrapper(torch.nn.Module, ABC):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, *args):
        """Defines the forward pass."""
        raise NotImplementedError

    @property
    @abstractmethod
    def atomic_numbers(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def atomic_energies(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def r_max(self):
        raise NotImplementedError

    @r_max.setter
    @abstractmethod
    def r_max(self, value):
        raise NotImplementedError


class MaceWrapper(AbstractWrapper):
    def __init__(self, args, model, optimize_atomic_energies=False):
        super().__init__(model)

        if optimize_atomic_energies:
            if 'atomic_energies' in self.model.atomic_energies_fn._buffers:
                atomic_energies = self.model.atomic_energies_fn.atomic_energies
                del self.model.atomic_energies_fn._buffers['atomic_energies']
                self.model.atomic_energies_fn.atomic_energies = torch.nn.Parameter(
                    atomic_energies
                )

        self.compute_force = args.forces_weight > 0.0
        self.compute_stress = args.stress_weight > 0.0

    def forward(self, *args):
        y_pred = self.model(
            *args,
            compute_force=self.compute_force,
            compute_stress=self.compute_stress,
            training=self.training,
        )

        if not isinstance(y_pred, dict):
            y_pred = {'energy': y_pred[0], 'forces': y_pred[1], 'stress': y_pred[2]}

        return y_pred

    @property
    def atomic_numbers(self):
        return AtomicNumberTable(self.model.atomic_numbers.cpu().tolist())

    @property
    def atomic_energies(self):
        return self.model.atomic_energies_fn.atomic_energies.cpu().tolist()

    @property
    def r_max(self):
        return self.model.r_max.item()

    @r_max.setter
    def r_max(self, r_max):
        if hasattr(self.model, 'radial_embedding'):
            if _HAS_MACE is False:
                raise ImportError(
                    "Optional dependency 'mace' is required for MaceWrapper."
                )

            num_bessel = self.model.radial_embedding.out_dim
            num_polynomial_cutoff = self.model.radial_embedding.cutoff_fn.p.item()

            if isinstance(self.model.radial_embedding.bessel_fn, BesselBasis):
                radial_type = 'bessel'
            elif isinstance(self.model.radial_embedding.bessel_fn, ChebychevBasis):
                radial_type = 'chebychev'
            elif isinstance(self.model.radial_embedding.bessel_fn, GaussianBasis):
                radial_type = 'gaussian'
            else:
                return

            if isinstance(
                self.model.radial_embedding.distance_transform, AgnesiTransform
            ):
                distance_transform = 'Agnesi'
            elif isinstance(
                self.model.radial_embedding.distance_transform, SoftTransform
            ):
                distance_transform = 'Soft'
            else:
                return

            self.model.radial_embedding = RadialEmbeddingBlock(
                r_max=r_max,
                num_bessel=num_bessel,
                num_polynomial_cutoff=num_polynomial_cutoff,
                radial_type=radial_type,
                distance_transform=distance_transform,
            )

        if hasattr(self.model, 'pair_repulsion'):
            if _HAS_MACE is False:
                raise ImportError(
                    "Optional dependency 'mace' is required for MaceWrapper."
                )

            if self.model.pair_repulsion:
                p = self.model.pair_repulsion_fn.p
                self.model.pair_repulsion_fn = ZBLBasis(r_max=r_max, p=p)

        self.model.r_max.fill_(r_max)


class SevennetWrapper(AbstractWrapper):
    def __init__(self, args, model):
        super().__init__(model)

    def forward(self, input):
        input.energy = input.y
        input.forces = input['force']
        input.edge_vec, _ = self.get_edge_vectors_and_lengths(
            input.positions, input.edge_index, input.shifts
        )
        input.num_atoms = input.ptr[1:] - input.ptr[:-1]

        y_pred = self.model(input)

        y_pred = {
            'energy': y_pred.inferred_total_energy,
            'forces': y_pred.inferred_force,
            'stress': self.batch_voigt_to_tensor(y_pred.inferred_stress).type(
                y_pred.inferred_total_energy.dtype
            ),
        }

        return y_pred

    @classmethod
    def get_edge_vectors_and_lengths(
        cls,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        shifts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        vectors = positions[receiver] - positions[sender] + shifts
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        return vectors, lengths

    @classmethod
    def batch_voigt_to_tensor(cls, voigts: torch.Tensor) -> torch.Tensor:
        tensors = torch.zeros(
            (voigts.shape[0], 3, 3), dtype=voigts.dtype, device=voigts.device
        )
        tensors[:, 0, 0] = voigts[:, 0]
        tensors[:, 1, 1] = voigts[:, 1]
        tensors[:, 2, 2] = voigts[:, 2]
        tensors[:, 1, 2] = tensors[:, 2, 1] = voigts[:, 3]
        tensors[:, 0, 2] = tensors[:, 2, 0] = voigts[:, 4]
        tensors[:, 0, 1] = tensors[:, 1, 0] = voigts[:, 5]
        return tensors

    @property
    def atomic_numbers(self):
        return AtomicNumberTable(
            torch.nonzero(self.model.z_to_onehot_tensor != -1).squeeze().cpu().tolist()
        )

    @property
    def atomic_energies(self):
        return None

    @property
    def r_max(self):
        return self.model.cutoff.item()

    @r_max.setter
    def r_max(self, value):
        self.model.cutoff.fill_(value)


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

        num_graphs = int(batch.max().item()) + 1
        data_list = []
        for graph_idx in range(num_graphs):
            mask = batch == graph_idx
            graph = Data(
                atomic_numbers=atomic_numbers[mask],
                positions=positions[mask],
                cell=lattice[graph_idx].to(self.model_dtype).unsqueeze(0)
                if lattice is not None
                else None,
                pbc=torch.zeros(3, dtype=torch.bool, device=positions.device),
            )
            graph.idx = graph_idx
            data_list.append(graph)

        return Batch.from_data_list(data_list)

    def _cast_batch_to_model_dtype(self, data):
        float_attrs = [
            'positions',
            'pos',
            'cell',
            'y',
            'force',
            'stress',
            'shifts',
            'unit_shifts',
            'cell_volume',
        ]
        for attr in float_attrs:
            value = getattr(data, attr, None)
            if torch.is_tensor(value) and value.is_floating_point():
                setattr(data, attr, value.to(self.model_dtype))
        return data

    def _data_to_atom_graphs(self, data):
        try:
            from ase import Atoms
        except ImportError as exc:
            raise ImportError(
                'ASE is required to construct ORB AtomGraphs. Install with: pip install ase'
            ) from exc

        from orb_models.forcefield.atomic_system import ase_atoms_to_atom_graphs

        positions = getattr(data, 'positions', getattr(data, 'pos', None))
        if positions is None:
            raise ValueError('Input data object must provide `positions`.')
        positions = positions.to(self.model_dtype)

        cell = getattr(data, 'cell', None)
        if cell is None:
            cell_tensor = torch.zeros(
                (3, 3), dtype=self.model_dtype, device=positions.device
            )
        else:
            cell_tensor = cell.to(self.model_dtype)
            if cell_tensor.dim() == 3:
                cell_tensor = cell_tensor[0]

        pbc = getattr(data, 'pbc', None)
        if pbc is None:
            pbc_tensor = torch.zeros(3, dtype=torch.bool, device=positions.device)
        else:
            pbc_tensor = pbc
            if pbc_tensor.dim() > 1:
                pbc_tensor = pbc_tensor[0]
            pbc_tensor = pbc_tensor.to(dtype=torch.bool)

        device = positions.device

        atoms = Atoms(
            numbers=data.atomic_numbers.detach().cpu().numpy(),
            positions=positions.detach().cpu().numpy(),
            cell=cell_tensor.detach().cpu().numpy(),
            pbc=pbc_tensor.detach().cpu().numpy(),
        )

        if hasattr(data, 'tags'):
            atoms.set_tags(data.tags.detach().cpu().numpy())

        system_id = int(getattr(data, 'idx', -1)) if hasattr(data, 'idx') else None

        return ase_atoms_to_atom_graphs(
            atoms,
            self.system_config,
            device=device,
            output_dtype=self.model_dtype,
            system_id=system_id,
        )

    def _as_atom_graphs(self, data):
        try:
            from torch_geometric.data import Batch
        except ImportError:
            Batch = None

        if Batch is not None and isinstance(data, Batch):
            data_list = data.to_data_list()
        else:
            data_list = [data]

        from orb_models.forcefield.base import batch_graphs

        atom_graphs = [self._data_to_atom_graphs(item) for item in data_list]
        if len(atom_graphs) == 1:
            return atom_graphs[0]
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


class AniWrapper(AbstractWrapper):
    """
    Wrapper for TorchANI models to be used with Equitrain.

    This wrapper integrates the Atomic Neural Network (ANI) potential from the TorchANI
    library into the Equitrain framework. It supports energy-only training workflows
    and can optionally compute forces using autograd.

    Parameters
    ----------
    args : object
        Arguments object containing training parameters
    model : torch.nn.Module, optional
        A pre-trained TorchANI model. If None, a new model will be created.
    species_order : List[str], optional
        List of chemical symbols in order (e.g., ['H', 'C', 'N', 'O']).
        Default is None, which will use the model's species order if available.
    """

    def __init__(self, args, model=None, species_order=None):
        """Initialize the ANI wrapper."""
        super().__init__(model)

        # Store arguments for later use
        self.compute_force = getattr(args, 'forces_weight', 0.0) > 0.0
        self.compute_stress = getattr(args, 'stress_weight', 0.0) > 0.0

        # If no model is provided, we need to create one
        if model is None:
            self.model = self._create_default_model(species_order)

        # Store species order for conversion
        self._species_order = species_order
        if self._species_order is None and hasattr(self.model, 'species_order'):
            self._species_order = self.model.species_order

        # Create species converter if needed
        try:
            import torchani

            if hasattr(self.model, 'species_converter'):
                self.species_converter = self.model.species_converter
            elif self._species_order is not None:
                self.species_converter = torchani.utils.ChemicalSymbolsToInts(
                    self._species_order
                )
        except ImportError:
            self.species_converter = None

        self._model_device = next(self.model.parameters()).device
        self._model_dtype = next(self.model.parameters()).dtype
        self._atomic_number_to_species_index = None
        if self._species_order is not None:
            try:
                from ase.data import atomic_numbers as _atomic_numbers_lookup
            except ImportError as exc:
                raise ImportError(
                    "Optional dependency 'ase' is required for ANI models."
                ) from exc

            self._atomic_number_to_species_index = {
                _atomic_numbers_lookup[symbol]: idx
                for idx, symbol in enumerate(self._species_order)
            }

    def _create_default_model(self, species_order=None):
        """
        Create a default ANI model.

        This is a fallback if no model is provided. It uses ANI-1x from the model zoo.

        Parameters
        ----------
        species_order : List[str], optional
            List of chemical symbols in order. Default is None, which will use
            the model's default species order.

        Returns
        -------
        torch.nn.Module
            A TorchANI model
        """
        try:
            import torchani

            # Use ANI-1x as default model
            model = torchani.models.ANI1x(periodic_table_index=False)
            model.species_order = ['H', 'C', 'N', 'O']
            return model
        except ImportError:
            raise ImportError(
                'TorchANI is required for ANI models. Install with: pip install torchani'
            )

    def _atomic_numbers_to_species_indices(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        if self._atomic_number_to_species_index is None:
            raise ValueError(
                'ANI wrapper requires a species order to convert atomic numbers to species indices.'
            )

        indices = []
        for value in atomic_numbers.detach().cpu().tolist():
            if value not in self._atomic_number_to_species_index:
                raise ValueError(
                    f'Observed atom type {value} that is not listed in the ANI species order.'
                )
            indices.append(self._atomic_number_to_species_index[value])

        return torch.tensor(indices, device=self._model_device, dtype=torch.long)

    def _prepare_batch_inputs(self, data):
        try:
            from torch_geometric.data import Batch
        except ImportError as exc:
            raise ImportError(
                'torch_geometric is required for ANI training workflows.'
            ) from exc

        if isinstance(data, Batch):
            molecules = data.to_data_list()
        else:
            molecules = [data]

        species_tensors: list[torch.Tensor] = []
        coordinate_tensors: list[torch.Tensor] = []
        counts: list[int] = []

        for molecule in molecules:
            coordinates = getattr(molecule, 'positions', getattr(molecule, 'pos', None))
            if coordinates is None:
                raise ValueError('ANI wrapper expects `positions` in the data object.')

            coordinates = coordinates.to(
                device=self._model_device, dtype=self._model_dtype
            )

            if hasattr(molecule, 'species'):
                species = molecule.species.to(self._model_device)
            else:
                atomic_numbers = getattr(molecule, 'atomic_numbers', None)
                if atomic_numbers is None:
                    raise ValueError(
                        'ANI wrapper requires either `species` or `atomic_numbers` in the data object.'
                    )
                species = self._atomic_numbers_to_species_indices(atomic_numbers)

            species_tensors.append(species)
            coordinate_tensors.append(coordinates)
            counts.append(species.shape[0])

        max_atoms = max(counts)
        batch_size = len(molecules)

        species_batch = torch.full(
            (batch_size, max_atoms),
            -1,
            dtype=torch.long,
            device=self._model_device,
        )
        coordinates_batch = torch.zeros(
            (batch_size, max_atoms, 3),
            dtype=self._model_dtype,
            device=self._model_device,
        )

        for index, (species, coords) in enumerate(zip(species_tensors, coordinate_tensors)):
            natoms = species.shape[0]
            species_batch[index, :natoms] = species
            coordinates_batch[index, :natoms] = coords

        coordinates_batch.requires_grad_(self.compute_force)
        return species_batch, coordinates_batch, counts

    def forward(self, *args):
        """
        Forward pass through the ANI model.

        Parameters
        ----------
        *args : tuple
            Input data. Can be a PyTorch Geometric Data object or a tuple of
            (species, coordinates).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing 'energy', 'forces', and 'stress' predictions.
        """
        if len(args) == 1:
            species_batch, coordinates_batch, counts = self._prepare_batch_inputs(args[0])
        else:
            species_batch = args[0].to(device=self._model_device, dtype=torch.long)
            coordinates_batch = args[1].to(
                device=self._model_device, dtype=self._model_dtype
            )
            coordinates_batch.requires_grad_(self.compute_force)
            counts = [
                int((species_batch[i] >= 0).sum().item())
                for i in range(species_batch.shape[0])
            ]

        outputs = self.model((species_batch, coordinates_batch))
        energies = getattr(outputs, 'energies', outputs[1]).to(self._model_dtype)
        energies = energies.view(-1)

        y_pred = {'energy': energies, 'forces': None, 'stress': None}

        if self.compute_force:
            forces_full = -torch.autograd.grad(
                energies.sum(),
                coordinates_batch,
                create_graph=True,
                retain_graph=self.compute_stress,
            )[0]
            force_chunks = [
                forces_full[index, :count] for index, count in enumerate(counts)
            ]
            y_pred['forces'] = (
                torch.cat(force_chunks, dim=0) if len(force_chunks) > 1 else force_chunks[0]
            )

        if self.compute_stress:
            batch_size = energies.shape[0]
            y_pred['stress'] = torch.zeros(
                batch_size, 3, 3, device=self._model_device, dtype=self._model_dtype
            )

        return y_pred

    @property
    def atomic_numbers(self):
        """
        Get the atomic numbers supported by the model.

        Returns
        -------
        AtomicNumberTable
            Table of atomic numbers supported by the model.
        """
        if hasattr(self.model, 'species_order'):
            # Convert chemical symbols to atomic numbers
            symbol_to_Z = {
                'H': 1,
                'He': 2,
                'Li': 3,
                'Be': 4,
                'B': 5,
                'C': 6,
                'N': 7,
                'O': 8,
                'F': 9,
                'Ne': 10,
                'Na': 11,
                'Mg': 12,
                'Al': 13,
                'Si': 14,
                'P': 15,
                'S': 16,
                'Cl': 17,
                'Ar': 18,
            }
            atomic_nums = [symbol_to_Z[symbol] for symbol in self.model.species_order]
            return AtomicNumberTable(atomic_nums)
        else:
            # Default to HCNO if no species_order is available
            return AtomicNumberTable([1, 6, 7, 8])

    @property
    def atomic_energies(self):
        """
        Get the atomic reference energies.

        Returns
        -------
        List[float] or None
            List of atomic reference energies or None if not available.
        """
        if hasattr(self.model, 'sae_dict'):
            # Return atomic energies if available
            return list(self.model.sae_dict.values())
        elif hasattr(self.model, 'energy_shifter') and hasattr(
            self.model.energy_shifter, 'self_energies'
        ):
            # Return self energies from energy shifter
            return self.model.energy_shifter.self_energies.tolist()
        else:
            # No atomic energies available
            return None

    @property
    def r_max(self):
        """
        Get the maximum cutoff radius.

        Returns
        -------
        float
            Maximum cutoff radius.
        """
        if hasattr(self.model, 'aev_computer'):
            return self.model.aev_computer.Rcr
        else:
            # Default cutoff for ANI-1x
            return 5.2

    @r_max.setter
    def r_max(self, value):
        """
        Set the maximum cutoff radius.

        Parameters
        ----------
        value : float
            New maximum cutoff radius.
        """
        # This is a no-op for ANI models as changing r_max would require
        # rebuilding the AEV computer, which would invalidate the model
        pass


__all__ = [
    'AbstractWrapper',
    'AniWrapper',
    'MaceWrapper',
    'OrbWrapper',
    'SevennetWrapper',
]
