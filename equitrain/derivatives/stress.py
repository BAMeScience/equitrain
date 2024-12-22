import torch

from typing import List, Optional, Tuple


def compute_stress(
    energy: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:

    grad_outputs: List[Optional[torch.Tensor]] = torch.ones_like(energy)

    virials = torch.autograd.grad(
        outputs      = energy,       # [n_graphs, ]
        inputs       = displacement, # [n_nodes, 3]
        grad_outputs = grad_outputs,
        retain_graph = training,       # Make sure the graph is not destroyed during training
        create_graph = training,       # Create graph for second derivative
        allow_unused = True,
    )[0]

    cell = cell.view(-1, 3, 3)
    volume = torch.einsum(
        "zi,zi->z",
        cell[:, 0, :],
        torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
    ).unsqueeze(-1)
    stress = virials / volume.view(-1, 1, 1)

    return stress


def get_displacement(
    positions: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor
) -> torch.Tensor:

    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype  = positions.dtype,
        device = positions.device,
    )
    displacement.requires_grad_(True)

    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )

    return positions, displacement
