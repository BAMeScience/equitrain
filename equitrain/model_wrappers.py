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
    Wrapper for ORB (Orbital Materials) models to be used with Equitrain.

    This wrapper integrates the ORB universal interatomic potential family from Orbital Materials
    into the Equitrain framework. ORB models are PyTorch-native and support energy, forces, and
    stress prediction workflows with excellent performance across >80 elements.

    Parameters
    ----------
    args : object
        Arguments object containing training parameters
    model : torch.nn.Module, optional
        A pre-trained ORB model. If None, a new model will be created.
    model_variant : str, optional
        ORB model variant ('direct' or 'conservative'). Default is 'direct'.
        - 'direct': forwards also output per-atom forces + stress
        - 'conservative': only energy; forces/stress via torch.autograd.grad
    enable_zbl : bool, optional
        Enable ZBL repulsion term for systems with high-Z elements (Z > 56). Default is False.
    """

    def __init__(self, args, model=None, model_variant='direct', enable_zbl=False):
        """Initialize the ORB wrapper."""
        super().__init__(model)

        # Store arguments for later use
        self.compute_force = getattr(args, 'forces_weight', 0.0) > 0.0
        self.compute_stress = getattr(args, 'stress_weight', 0.0) > 0.0
        self.model_variant = model_variant
        self.enable_zbl = enable_zbl

        # If no model is provided, create a default one
        if model is None:
            self.model = self._create_default_model()

        # Cache for graph compilation - run dummy batch to avoid first-step delay
        self._compile_cache_initialized = False
        self._initialize_compile_cache()

    def _create_default_model(self):
        """
        Create a default ORB model from the model zoo.

        Returns
        -------
        torch.nn.Module
            An ORB model from the OMat24 model zoo
        """
        try:
            from orb_models.forcefield import pretrained

            # Use OMat24 model from the zoo (covers >80 elements)
            if self.model_variant == 'direct':
                model = pretrained.orb_v3_small_direct()
            else:
                model = pretrained.orb_v3_small()

            # Enable ZBL repulsion if requested
            if self.enable_zbl:
                model.enable_zbl = True

            return model
        except ImportError:
            raise ImportError(
                'ORB models are required. Install with: pip install "orb-models>=3.0"'
            )

    def _initialize_compile_cache(self):
        """
        Initialize graph compilation cache by running dummy batches.

        This prevents compilation delays on the first forward pass by pre-compiling
        common batch sizes with torch.compile.
        """
        if self._compile_cache_initialized:
            return

        try:
            # Create dummy batches for common sizes to trigger compilation
            dummy_sizes = [1, 8, 16, 32]
            device = next(self.model.parameters()).device

            for batch_size in dummy_sizes:
                # Create dummy input
                dummy_atomic_numbers = torch.randint(
                    1, 84, (batch_size * 10,), device=device
                )
                dummy_positions = torch.randn(batch_size * 10, 3, device=device)
                dummy_lattice = (
                    torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
                )

                # Create batch indices
                dummy_batch = torch.repeat_interleave(
                    torch.arange(batch_size, device=device), 10
                )

                with torch.no_grad():
                    try:
                        _ = self.model(
                            atomic_numbers=dummy_atomic_numbers,
                            positions=dummy_positions,
                            lattice=dummy_lattice,
                            batch=dummy_batch,
                        )
                    except Exception:
                        # If dummy forward fails, continue - real data might work
                        pass

            self._compile_cache_initialized = True
        except Exception:
            # If cache initialization fails, continue without it
            self._compile_cache_initialized = True

    def forward(self, *args):
        """
        Forward pass through the ORB model.

        Parameters
        ----------
        *args : tuple
            Input data. Can be a PyTorch Geometric Data object or a tuple of
            (atomic_numbers, positions, lattice, batch).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing 'energy', 'forces', and 'stress' predictions.
        """
        # Handle different input formats
        if len(args) == 1:
            # PyG Data object
            data = args[0]
            atomic_numbers = data.atomic_numbers
            positions = data.positions
            lattice = getattr(data, 'cell', None)
            batch = getattr(data, 'batch', None)

            # If no lattice provided, create identity matrices for non-periodic systems
            if lattice is None:
                batch_size = batch.max().item() + 1 if batch is not None else 1
                lattice = (
                    torch.eye(3, device=positions.device)
                    .unsqueeze(0)
                    .repeat(batch_size, 1, 1)
                )

        else:
            # Direct arguments
            atomic_numbers = args[0]
            positions = args[1]
            lattice = args[2] if len(args) > 2 else None
            batch = args[3] if len(args) > 3 else None

            # Default lattice if not provided
            if lattice is None:
                batch_size = batch.max().item() + 1 if batch is not None else 1
                lattice = (
                    torch.eye(3, device=positions.device)
                    .unsqueeze(0)
                    .repeat(batch_size, 1, 1)
                )

        # Enable mixed precision for better performance
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            if self.model_variant == 'direct':
                # Direct variant outputs energy, forces, and stress directly
                result = self.model(
                    atomic_numbers=atomic_numbers,
                    positions=positions,
                    lattice=lattice,
                    batch=batch,
                )

                # ORB direct models return a dictionary with energy, forces, stress
                y_pred = {
                    'energy': result.get('energy', result.get('total_energy')),
                    'forces': result.get('forces') if self.compute_force else None,
                    'stress': result.get('stress') if self.compute_stress else None,
                }

            else:
                # Conservative variant - only energy, compute forces/stress via autograd
                positions_grad = positions.requires_grad_(
                    self.compute_force or self.compute_stress
                )
                lattice_grad = (
                    lattice.requires_grad_(self.compute_stress)
                    if self.compute_stress
                    else lattice
                )

                energy = self.model(
                    atomic_numbers=atomic_numbers,
                    positions=positions_grad,
                    lattice=lattice_grad,
                    batch=batch,
                )

                y_pred = {'energy': energy, 'forces': None, 'stress': None}

                # Compute forces via autograd if needed
                if self.compute_force:
                    forces = -torch.autograd.grad(
                        energy.sum(),
                        positions_grad,
                        create_graph=self.compute_stress,
                        retain_graph=self.compute_stress,
                    )[0]
                    y_pred['forces'] = forces

                # Compute stress via autograd if needed
                if self.compute_stress:
                    # Stress computation via lattice gradients
                    stress_grad = torch.autograd.grad(
                        energy.sum(),
                        lattice_grad,
                        create_graph=False,
                        retain_graph=False,
                    )[0]

                    # Convert lattice gradients to stress tensor
                    volume = torch.det(lattice_grad)
                    stress = stress_grad / volume.unsqueeze(-1).unsqueeze(-1)
                    y_pred['stress'] = stress

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
        # ORB models support elements 1-84 (H to Po)
        # Return the full range as ORB is a universal potential
        if hasattr(self.model, 'atomic_numbers'):
            return AtomicNumberTable(self.model.atomic_numbers.cpu().tolist())
        else:
            # Default to elements 1-84 for ORB universal models
            return AtomicNumberTable(list(range(1, 85)))

    @property
    def atomic_energies(self):
        """
        Get the atomic reference energies.

        Returns
        -------
        List[float] or None
            List of atomic reference energies or None if not available.
        """
        # ORB models typically don't expose atomic reference energies
        # as they are learned during training
        if hasattr(self.model, 'atomic_energies'):
            return self.model.atomic_energies.cpu().tolist()
        else:
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
        if hasattr(self.model, 'cutoff'):
            return (
                self.model.cutoff.item()
                if torch.is_tensor(self.model.cutoff)
                else self.model.cutoff
            )
        elif hasattr(self.model, 'r_max'):
            return (
                self.model.r_max.item()
                if torch.is_tensor(self.model.r_max)
                else self.model.r_max
            )
        else:
            # Default cutoff for ORB models (typically around 6.0 Å)
            return 6.0

    @r_max.setter
    def r_max(self, value):
        """
        Set the maximum cutoff radius.

        Parameters
        ----------
        value : float
            New maximum cutoff radius.
        """
        if hasattr(self.model, 'cutoff'):
            if torch.is_tensor(self.model.cutoff):
                self.model.cutoff.fill_(value)
            else:
                self.model.cutoff = value
        elif hasattr(self.model, 'r_max'):
            if torch.is_tensor(self.model.r_max):
                self.model.r_max.fill_(value)
            else:
                self.model.r_max = value
        # Note: Changing r_max for ORB models may require model recompilation
        # depending on the specific architecture
