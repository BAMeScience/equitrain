from __future__ import annotations

from equitrain import get_args_parser_train
from equitrain.backends import jax_checkpoint, torch_checkpoint


def _make_checkpoint_dirs(base):
    for name in (
        'best_val_epochs@1_e@0.5',
        'best_val_epochs@2_e@0.2',
        'best_val_epochs@3_e@0.3',
        'best_val_epochs@4_e@0.1',
        'best_val_epochs@5_e@1e-05',
    ):
        (base / name).mkdir()


def _remaining_checkpoint_dirs(base):
    return sorted(path.name for path in base.glob('best_val_epochs@*_e@*'))


def test_keep_best_checkpoints_train_arg():
    args = get_args_parser_train().parse_args(['--keep-best-checkpoints', '2'])

    assert args.keep_best_checkpoints == 2


def test_torch_prune_best_checkpoints_keeps_lowest_validation_losses(tmp_path):
    _make_checkpoint_dirs(tmp_path)

    torch_checkpoint._prune_best_checkpoints(tmp_path, 'val', 2)

    assert _remaining_checkpoint_dirs(tmp_path) == [
        'best_val_epochs@4_e@0.1',
        'best_val_epochs@5_e@1e-05',
    ]


def test_jax_prune_best_checkpoints_keeps_lowest_validation_losses(tmp_path):
    _make_checkpoint_dirs(tmp_path)

    jax_checkpoint._prune_best_checkpoints(tmp_path, 'val', 2)

    assert _remaining_checkpoint_dirs(tmp_path) == [
        'best_val_epochs@4_e@0.1',
        'best_val_epochs@5_e@1e-05',
    ]


def test_prune_best_checkpoints_default_keeps_all(tmp_path):
    _make_checkpoint_dirs(tmp_path)

    torch_checkpoint._prune_best_checkpoints(tmp_path, 'val', 0)

    assert _remaining_checkpoint_dirs(tmp_path) == [
        'best_val_epochs@1_e@0.5',
        'best_val_epochs@2_e@0.2',
        'best_val_epochs@3_e@0.3',
        'best_val_epochs@4_e@0.1',
        'best_val_epochs@5_e@1e-05',
    ]
