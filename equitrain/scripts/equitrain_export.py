import json
import sys
from pathlib import Path

from .. import check_args_complete, get_args_parser_export
from ..argparser import ArgsFormatter
from ..backends.torch_checkpoint import (
    _find_best_checkpoint,
    _find_last_checkpoint,
    _load_state_dict,
    _resolve_model_path,
)
from ..backends.torch_model import get_model
from ..backends.torch_wrappers import AbstractWrapper
from ..checkpoint import load_checkpoint
from ..logger import FileLogger


def _strip_module_prefix(name: str) -> str:
    return name.replace('module.', '', 1) if name.startswith('module.') else name


def _resolve_checkpoint_directory(args) -> Path | None:
    load_checkpoint = getattr(args, 'load_checkpoint', None)
    load_checkpoint_model = getattr(args, 'load_checkpoint_model', None)

    if load_checkpoint is None and getattr(args, 'load_best_checkpoint', False):
        load_checkpoint, _ = _find_best_checkpoint(args.output_dir, 'val')

    if load_checkpoint is None and getattr(args, 'load_last_checkpoint', False):
        load_checkpoint, _ = _find_last_checkpoint(args.output_dir, 'val')

    if load_checkpoint is not None:
        return Path(load_checkpoint)

    if load_checkpoint_model is None and getattr(
        args, 'load_best_checkpoint_model', False
    ):
        checkpoint_dir, _ = _find_best_checkpoint(args.output_dir, 'val')
        if checkpoint_dir is not None:
            return Path(checkpoint_dir)

    if load_checkpoint_model is None and getattr(
        args, 'load_last_checkpoint_model', False
    ):
        checkpoint_dir, _ = _find_last_checkpoint(args.output_dir, 'val')
        if checkpoint_dir is not None:
            return Path(checkpoint_dir)

    if load_checkpoint_model is not None:
        return Path(load_checkpoint_model).parent

    return None


def _resolve_checkpoint_model_path(args) -> str | None:
    checkpoint_dir = _resolve_checkpoint_directory(args)
    if checkpoint_dir is None:
        return None

    direct_model = getattr(args, 'load_checkpoint_model', None)
    if direct_model is not None and Path(direct_model).parent == checkpoint_dir:
        return direct_model

    return _resolve_model_path(checkpoint_dir)


def _load_checkpoint_state_dict(args):
    model_path = _resolve_checkpoint_model_path(args)
    if model_path is None:
        return None
    return _load_state_dict(Path(model_path), map_location='cpu')


def _checkpoint_fine_tune_config(args) -> dict | None:
    checkpoint_dir = _resolve_checkpoint_directory(args)
    if checkpoint_dir is None:
        return None
    args_path = checkpoint_dir / 'args.json'
    if not args_path.exists():
        return None
    payload = json.loads(args_path.read_text())
    config = payload.get('fine_tune_export')
    return config if isinstance(config, dict) else None


def _checkpoint_adapter_keys(state_dict) -> set[str]:
    if state_dict is None:
        return set()

    keys = {_strip_module_prefix(str(key)) for key in state_dict}
    adapters = set()
    if any(
        key.startswith('_lora_a_params.') or key.startswith('_lora_b_params.')
        for key in keys
    ):
        adapters.add('lora')
    if any(key.startswith('_delta_params.') for key in keys):
        adapters.add('delta')
    return adapters


def _detect_fine_tune_wrapper(state_dict, config: dict | None) -> str | None:
    if config is not None:
        wrapper_name = config.get('wrapper')
        if wrapper_name in {'delta', 'lora'}:
            return str(wrapper_name)

    adapters = _checkpoint_adapter_keys(state_dict)
    if adapters:
        adapter_list = ', '.join(sorted(adapters))
        raise ValueError(
            'Checkpoint contains fine-tune adapter parameters '
            f'({adapter_list}) but no fine_tune_export metadata. '
            'Re-train the adapter checkpoint with the current Equitrain version '
            'before using equitrain-export.'
        )
    return None


def _lora_kwargs(config: dict | None) -> dict:
    config = config or {}
    return {
        'rank_fraction': config.get('rank_fraction'),
        'rank_reduction': config.get('rank_reduction'),
        'min_rank': config.get('min_rank', 1),
        'alpha': config.get('alpha'),
    }


def _wrap_for_fine_tune_export(args, model, state_dict, logger):
    requested = getattr(args, 'fine_tune_wrapper', 'auto') or 'auto'
    config = _checkpoint_fine_tune_config(args)
    detected = _detect_fine_tune_wrapper(state_dict, config)

    if requested == 'none':
        if detected is not None:
            logger.log(
                1,
                'Checkpoint contains fine-tune adapter parameters, but '
                '--fine-tune-wrapper=none was requested. Exporting base model only.',
            )
        return model

    if requested == 'auto':
        if detected is None:
            return model
        requested = detected
        logger.log(1, f'Detected {requested} fine-tune wrapper in checkpoint')
    elif detected is not None and detected != requested:
        raise ValueError(
            f'Checkpoint contains {detected} adapter parameters, but '
            f'--fine-tune-wrapper={requested} was requested.'
        )

    if requested == 'delta':
        from equitrain.finetune.delta_torch import DeltaFineTuneWrapper

        logger.log(1, 'Wrapping base model with delta fine-tune adapter for export')
        return DeltaFineTuneWrapper(model)

    if requested == 'lora':
        if config is None:
            raise ValueError(
                'Exporting a LoRA checkpoint requires fine_tune_export metadata '
                'saved by the current Equitrain checkpoint writer.'
            )

        from equitrain.finetune.lora_torch import LoRAFineTuneWrapper

        kwargs = _lora_kwargs(config)
        logger.log(1, 'Wrapping base model with LoRA fine-tune adapter for export')
        return LoRAFineTuneWrapper(model, **kwargs)

    return model


def _save_exported_model(model, filename):
    import torch

    model = model.cpu()

    export_fn = getattr(model, 'export', None)
    if callable(export_fn):
        export_fn(filename)
        return

    if isinstance(model, AbstractWrapper):
        model = model.model

    torch.save(model.cpu(), filename)


# %%
def _export(args):
    logger = FileLogger(
        log_to_file=True,
        enable_logging=True,
        verbosity=args.verbose,
    )
    logger.log(1, ArgsFormatter(args))

    model = get_model(args)
    state_dict = _load_checkpoint_state_dict(args)
    model = _wrap_for_fine_tune_export(args, model, state_dict, logger)

    # Import model weights from a checkpoint if requested.
    if not load_checkpoint(args, model, logger=logger)[0]:
        logger.log(1, 'No checkpoint found, using initial model')

    _save_exported_model(model, args.model_export)


# %%
def export(args):
    check_args_complete(args, 'export')

    if args.model is None:
        raise ValueError('--model is a required argument')
    if args.model_export is None:
        raise ValueError('--model-export is a required argument')

    _export(args)


# %%
def main():
    parser = get_args_parser_export()

    try:
        export(parser.parse_args())

    except ValueError as v:
        print(v, file=sys.stderr)
        sys.exit(1)


# %%
if __name__ == '__main__':
    main()
