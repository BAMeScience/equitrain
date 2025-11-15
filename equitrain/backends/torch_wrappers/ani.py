"""
TorchANI wrapper.
"""

from __future__ import annotations

import torch

from equitrain.data.atomic import AtomicNumberTable

from .base import AbstractWrapper


class AniWrapper(AbstractWrapper):
    """
    Wrapper for TorchANI models to be used with Equitrain.

    This wrapper integrates the Atomic Neural Network (ANI) potential from the TorchANI
    library into the Equitrain framework. It supports energy-only training workflows
    and can optionally compute forces using autograd.
    """

    def __init__(self, args, model=None, species_order=None):
        super().__init__(model)

        self.compute_force = getattr(args, 'forces_weight', 0.0) > 0.0
        self.compute_stress = getattr(args, 'stress_weight', 0.0) > 0.0

        if model is None:
            self.model = self._create_default_model(species_order)

        self._species_order = species_order
        if self._species_order is None and hasattr(self.model, 'species_order'):
            self._species_order = self.model.species_order

        try:
            from torchani import SpeciesConverter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                'TorchANI is required for AniWrapper. Install with `pip install torchani`.'
            ) from exc

        self._converter = SpeciesConverter(self._species_order)
        self._model_device = next(self.model.parameters()).device
        self._model_dtype = next(self.model.parameters()).dtype

    def _create_default_model(self, species_order):
        try:
            import torchani
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                'TorchANI is required for AniWrapper. Install with `pip install torchani`.'
            ) from exc

        if species_order is None:
            return torchani.models.ANI1x()

        return torchani.ANIModel(torchani.AEVComputer(species_order))

    def _atomic_numbers_to_species_indices(self, atomic_numbers):
        if self._species_order is None:
            raise ValueError(
                'Species order is required to convert atomic numbers to indices.'
            )
        species_to_index = {
            symbol: idx for idx, symbol in enumerate(self._species_order)
        }

        if torch.is_tensor(atomic_numbers):
            numbers = atomic_numbers.detach().cpu().tolist()
        else:
            numbers = list(atomic_numbers)

        try:
            from ase.data import chemical_symbols
        except (
            Exception
        ) as exc:  # pragma: no cover - ase is a core dependency for tests
            raise ImportError(
                'ase is required to map atomic numbers to symbols for ANI models.'
            ) from exc

        indices = []
        for number in numbers:
            symbol = chemical_symbols[int(number)]
            idx = species_to_index.get(symbol)
            if idx is None:
                break
            indices.append(idx)

        if len(indices) != len(numbers):
            raise ValueError(
                'Mismatch between provided atomic numbers and model species order.'
            )
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

        for index, (species, coords) in enumerate(
            zip(species_tensors, coordinate_tensors, strict=True)
        ):
            natoms = species.shape[0]
            species_batch[index, :natoms] = species
            coordinates_batch[index, :natoms] = coords

        coordinates_batch.requires_grad_(self.compute_force)
        return species_batch, coordinates_batch, counts

    def forward(self, *args):
        if len(args) == 1:
            species_batch, coordinates_batch, counts = self._prepare_batch_inputs(
                args[0]
            )
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
                torch.cat(force_chunks, dim=0)
                if len(force_chunks) > 1
                else force_chunks[0]
            )

        if self.compute_stress:
            batch_size = energies.shape[0]
            y_pred['stress'] = torch.zeros(
                batch_size, 3, 3, device=self._model_device, dtype=self._model_dtype
            )

        return y_pred

    @property
    def atomic_numbers(self):
        if hasattr(self.model, 'species_order'):
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
            }
            numbers = sorted(
                {symbol_to_Z.get(symbol, symbol) for symbol in self.model.species_order}
            )
            return AtomicNumberTable(numbers)

        return AtomicNumberTable([1, 6, 7, 8])

    @property
    def atomic_energies(self):
        if hasattr(self.model, 'sae_dict'):
            return list(self.model.sae_dict.values())
        if hasattr(self.model, 'energy_shifter') and hasattr(
            self.model.energy_shifter, 'self_energies'
        ):
            return self.model.energy_shifter.self_energies.tolist()
        return None

    @property
    def r_max(self):
        if hasattr(self.model, 'aev_computer'):
            return self.model.aev_computer.Rcr
        return 5.2

    @r_max.setter
    def r_max(self, value):
        # This is a no-op for ANI models; changing r_max would require rebuilding the AEV computer.
        return


__all__ = ['AniWrapper']
