import torch

from typing import List, Optional, Tuple

def compute_force(
    energy: torch.Tensor,
    positions: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:

    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]

    gradient = torch.autograd.grad(
        outputs=energy,  # [n_graphs, ]
        inputs=positions,  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)

    return -1 * gradient
