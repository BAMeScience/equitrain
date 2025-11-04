from abc import ABC, abstractmethod

import torch

from equitrain.data.atomic import AtomicNumberTable


class AbstractWrapper(torch.nn.Module, ABC):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @abstractmethod
    def forward(self, *args):
        """
        Defines the forward pass. Should implement the forward pass for the model.
        """
        pass

    @property
    @abstractmethod
    def atomic_numbers(self):
        """
        Property that should return atomic numbers from the model.
        """
        pass

    @property
    @abstractmethod
    def atomic_energies(self):
        """
        Property that should return atomic numbers from the model.
        """
        pass

    @property
    @abstractmethod
    def r_max(self):
        """
        Property that should return the r_max value from the model.
        """
        pass

    @r_max.setter
    @abstractmethod
    def r_max(self, value):
        """
        Setter for r_max. Should modify the model's r_max.
        """
        pass


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
        return AtomicNumberTable(self.model.atomic_numbers.tolist())

    @property
    def atomic_energies(self):
        return self.model.atomic_energies_fn.atomic_energies.tolist()

    @property
    def r_max(self):
        return self.model.r_max.item()

    @r_max.setter
    def r_max(self, r_max):
        if hasattr(self.model, 'radial_embedding'):
            from mace.modules.blocks import RadialEmbeddingBlock
            from mace.modules.radial import (
                AgnesiTransform,
                BesselBasis,
                ChebychevBasis,
                GaussianBasis,
                SoftTransform,
            )

            num_bessel = self.model.radial_embedding.out_dim
            num_polynomial_cutoff = self.model.radial_embedding.cutoff_fn.p.item()

            if isinstance(self.model.radial_embedding.bessel_fn, BesselBasis):
                radial_type = 'bessel'
            elif isinstance(self.model.radial_embedding.bessel_fn, ChebychevBasis):
                radial_type = 'chebychev'
            elif isinstance(self.model.radial_embedding.bessel_fn, GaussianBasis):
                radial_type = 'gaussian'
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
            from mace.modules.radial import ZBLBasis

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
        positions: torch.Tensor,  # [n_nodes, 3]
        edge_index: torch.Tensor,  # [2, n_edges]
        shifts: torch.Tensor,  # [n_edges, 3]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]

        return vectors, lengths

    @classmethod
    def batch_voigt_to_tensor(cls, voigts):
        """
        Convert a batch of Voigt notation arrays back to 3x3 stress tensors.

        Parameters:
            voigts (torch.Tensor): Tensor of shape (N, 6) representing N Voigt stress vectors.
                                Gradients will be preserved if attached.

        Returns:
            torch.Tensor: Tensor of shape (N, 3, 3) with full stress tensors.
        """
        tensors = torch.zeros(
            (voigts.shape[0], 3, 3), dtype=voigts.dtype, device=voigts.device
        )
        tensors[:, 0, 0] = voigts[:, 0]  # σ_xx
        tensors[:, 1, 1] = voigts[:, 1]  # σ_yy
        tensors[:, 2, 2] = voigts[:, 2]  # σ_zz
        tensors[:, 1, 2] = tensors[:, 2, 1] = voigts[:, 3]  # σ_yz
        tensors[:, 0, 2] = tensors[:, 2, 0] = voigts[:, 4]  # σ_xz
        tensors[:, 0, 1] = tensors[:, 1, 0] = voigts[:, 5]  # σ_xy
        return tensors

    @property
    def atomic_numbers(self):
        return AtomicNumberTable(
            torch.nonzero(self.model.z_to_onehot_tensor != -1).squeeze().tolist()
        )

    @property
    def atomic_energies(self):
        return None

    @property
    def r_max(self):
        return self.model.cutoff

    @r_max.setter
    def r_max(self, value):
        self.model.cutoff.fill_(value)


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

    def _atomic_numbers_to_species_indices(
        self, atomic_numbers: torch.Tensor
    ) -> torch.Tensor:
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

        for index, (species, coords) in enumerate(
            zip(species_tensors, coordinate_tensors)
        ):
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
