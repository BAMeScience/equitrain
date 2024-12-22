
import numpy as np
import torch

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
