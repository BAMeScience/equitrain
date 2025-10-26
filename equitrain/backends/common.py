from __future__ import annotations

from pathlib import Path

from equitrain.argparser import ArgumentError, check_args_complete
from equitrain.logger import FileLogger


def validate_training_args(
    args,
    backend_name: str,
    *,
    require_validation_file: bool = True,
) -> None:
    """
    Perform backend-agnostic validation of mandatory training arguments.
    """
    check_args_complete(args, 'train')

    missing: list[str] = []
    if getattr(args, 'train_file', None) is None:
        missing.append('--train-file')
    if require_validation_file and getattr(args, 'valid_file', None) is None:
        missing.append('--valid-file')
    if getattr(args, 'output_dir', None) is None:
        missing.append('--output-dir')
    if getattr(args, 'model', None) is None:
        missing.append('--model')

    if missing:
        raise ArgumentError(
            f'{backend_name} backend requires the following arguments: {", ".join(missing)}'
        )

    _ensure_losses_defined(args, backend_name)


def validate_evaluate_args(args, backend_name: str) -> None:
    """Backend-agnostic validation for evaluation scripts."""
    check_args_complete(args, 'evaluate')

    missing: list[str] = []
    if getattr(args, 'test_file', None) is None:
        missing.append('--test-file')
    if getattr(args, 'model', None) is None:
        missing.append('--model')

    if missing:
        raise ArgumentError(
            f'{backend_name} backend requires the following arguments: {", ".join(missing)}'
        )

    _ensure_losses_defined(args, backend_name)


def _ensure_losses_defined(args, backend_name: str) -> None:
    energy = getattr(args, 'energy_weight', 0.0) or 0.0
    forces = getattr(args, 'forces_weight', 0.0) or 0.0
    stress = getattr(args, 'stress_weight', 0.0) or 0.0
    if energy == 0.0 and forces == 0.0 and stress == 0.0:
        raise ArgumentError(
            f'{backend_name} backend requires at least one non-zero loss weight.'
        )


def ensure_output_dir(path: str | None) -> None:
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def init_logger(
    args,
    *,
    backend_name: str,
    enable_logging: bool,
    log_to_file: bool,
    output_dir: str | None,
) -> FileLogger:
    """Create a logger with consistent naming and verbosity."""
    return FileLogger(
        enable_logging=enable_logging,
        log_to_file=log_to_file,
        output_dir=output_dir,
        logger_name=f'Equitrain[{backend_name}]',
        verbosity=getattr(args, 'verbose', 0),
    )
