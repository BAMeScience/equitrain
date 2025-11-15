import torch


def compute_force(
    energy: torch.Tensor,
    positions: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    grad_outputs = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=energy,
        inputs=positions,
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
    )[0]
    if gradient is None:
        return torch.zeros_like(positions)
    return -gradient


__all__ = ['compute_force']
