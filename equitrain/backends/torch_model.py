from __future__ import annotations

import math

import torch

from . import torch_wrappers as _torch_wrappers


def get_model(args, logger=None):
    if isinstance(args.model, torch.nn.Module):
        model = args.model
    else:
        model = torch.load(args.model, weights_only=False)

    AbstractWrapper = _torch_wrappers.AbstractWrapper

    if not isinstance(model, AbstractWrapper):
        if not hasattr(args, 'energy_weight'):
            setattr(args, 'energy_weight', 0.0)
        if not hasattr(args, 'forces_weight'):
            setattr(args, 'forces_weight', 0.0)
        if not hasattr(args, 'stress_weight'):
            setattr(args, 'stress_weight', 0.0)

        wrapper_name = str(getattr(args, 'model_wrapper', '') or '').strip().lower()
        if wrapper_name == 'ani':
            model = _torch_wrappers.AniWrapper(args, model)
        elif wrapper_name == 'mace':
            model = _torch_wrappers.MaceWrapper(args, model)
        elif wrapper_name == 'orb':
            model = _torch_wrappers.OrbWrapper(args, model)
        elif wrapper_name == 'sevennet':
            model = _torch_wrappers.SevennetWrapper(args, model)
        elif wrapper_name == 'm3gnet':
            model = _torch_wrappers.M3GNetWrapper(args, model)
        else:
            raise ValueError(
                f"Unsupported torch model_wrapper '{wrapper_name}'. "
                'Use one of: mace, ani, orb, sevennet, m3gnet.'
            )

    if hasattr(args, 'r_max') and args.r_max is not None:
        requested_r_max = float(args.r_max)
        current_r_max = getattr(model, 'r_max', None)
        same_r_max = False
        if current_r_max is not None:
            same_r_max = math.isclose(
                float(current_r_max),
                requested_r_max,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        if same_r_max:
            if logger is not None:
                logger.log(1, f'Keeping existing r_max={requested_r_max}')
        else:
            if logger is not None:
                current_text = (
                    'unknown' if current_r_max is None else f'{float(current_r_max)}'
                )
                logger.log(
                    1,
                    'Overriding r_max from '
                    f'{current_text} to {requested_r_max}. This changes '
                    'cutoff-dependent model parameters.',
                )
            model.r_max = requested_r_max

    return model


__all__ = ['get_model']
