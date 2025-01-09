import random

import numpy as np
import torch

from equitrain.argparser import ArgumentError


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def set_dtype(dtype_str: str) -> None:
    match dtype_str.lower():
        case 'float16':
            torch.set_default_dtype(torch.float16)
        case 'float32':
            torch.set_default_dtype(torch.float32)
        case 'float64':
            torch.set_default_dtype(torch.float64)
        case _:
            raise ArgumentError('invalid dtype')
