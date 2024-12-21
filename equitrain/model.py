import torch

from equitrain.model_wrappers import *

from equitrain.equiformer_v1 import DotProductAttentionTransformerOC20
from equitrain.equiformer_v2 import EquiformerV2_OC20


def get_model(r_max, args, compute_force=True, compute_stress=True, logger=None):

    if isinstance(args.model, torch.nn.Module):

        model = args.model

    else:

        if args.model == "v1":
            model = DotProductAttentionTransformerOC20(
                # First three arguments are not used
                None, None, None,
                compute_forces   = compute_force,
                compute_stress   = compute_stress,
                max_radius       = r_max,
                max_num_elements = 95,
                alpha_drop       = args.alpha_drop,
                proj_drop        = args.proj_drop,
                drop_path_rate   = args.drop_path_rate,
                out_drop         = args.out_drop,
            )
        elif args.model == "v2":
            model = EquiformerV2_OC20(
                # First three arguments are not used
                None, None, None,
                compute_forces   = compute_force,
                compute_stress   = compute_stress,
                max_radius       = r_max,
                max_num_elements = 95,
                alpha_drop       = args.alpha_drop,
                drop_path_rate   = args.drop_path_rate,
                proj_drop        = args.proj_drop,
            )
        else:
            model = torch.load(args.model)

    if args.load_checkpoint_model is not None:

        if logger is not None:
            logger.info(f'Loading model checkpoint {args.load_checkpoint_model}...')

        model.load_state_dict(torch.load(args.load_checkpoint_model))

    if args.model_wrapper == "mace":
        model = MaceWrapper(model)

    return model
