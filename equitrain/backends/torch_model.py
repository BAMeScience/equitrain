from __future__ import annotations

import torch

from .torch_wrappers import (
    AbstractWrapper,
    AniWrapper,
    MaceWrapper,
    OrbWrapper,
    SevennetWrapper,
)


def get_model(args, logger=None):
    if isinstance(args.model, torch.nn.Module):
        model = args.model
    else:
        model = torch.load(args.model, weights_only=False)

    if not isinstance(model, AbstractWrapper):
        if not hasattr(args, 'energy_weight'):
            setattr(args, 'energy_weight', 0.0)
        if not hasattr(args, 'forces_weight'):
            setattr(args, 'forces_weight', 0.0)
        if not hasattr(args, 'stress_weight'):
            setattr(args, 'stress_weight', 0.0)

        if args.model_wrapper == 'ani':
            model = AniWrapper(args, model)
        if args.model_wrapper == 'mace':
            model = MaceWrapper(args, model)
        if args.model_wrapper == 'orb':
            model = OrbWrapper(args, model)
        if args.model_wrapper == 'sevennet':
            model = SevennetWrapper(args, model)

    if hasattr(args, 'r_max') and args.r_max is not None:
        if logger is not None:
            logger.log(1, f'Overwriting r_max model parameter with r_max={args.r_max}')
        model.r_max = args.r_max

    return model


__all__ = ['get_model']
