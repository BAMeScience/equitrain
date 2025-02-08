import torch


class MaceWrapper(torch.nn.Module):
    def __init__(self, args, model):
        super().__init__()

        self.model = model
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


class SevennetWrapper(torch.nn.Module):
    def __init__(self, args, model):
        super().__init__()

        self.model = model
        self.compute_force = args.forces_weight > 0.0
        self.compute_stress = args.stress_weight > 0.0

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
            'stress': self.batch_voigt_to_tensor(y_pred.inferred_stress),
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
