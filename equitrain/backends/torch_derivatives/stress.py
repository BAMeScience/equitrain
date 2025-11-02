import torch


def compute_stress(
    energy: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    grad_outputs = torch.ones_like(energy)
    virials = torch.autograd.grad(
        outputs=energy,
        inputs=displacement,
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
    )[0]

    cell = cell.view(-1, 3, 3)
    volume = torch.einsum(
        'zi,zi->z',
        cell[:, 0, :],
        torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
    ).unsqueeze(-1)
    return virials / volume.view(-1, 1, 1)


def get_displacement(
    positions: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)

    symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
    positions = positions + torch.einsum(
        'be,bec->bc', positions, symmetric_displacement[batch]
    )

    return positions, displacement


__all__ = ['compute_stress', 'get_displacement']
