from types import SimpleNamespace

import pytest

from equitrain.argparser import (
    check_args_complete,
    get_args_parser,
    get_args_parser_train,
)
from equitrain.backends.jax_scheduler import create_scheduler_controller
from equitrain.backends.scheduler_common import scheduler_kwargs


def test_decay_rate_alias_sets_gamma_without_adding_decay_rate():
    args = get_args_parser_train().parse_args([])
    assert args.gamma == 0.8
    assert not hasattr(args, 'decay_rate')
    assert scheduler_kwargs(args)['gamma'] == 0.8
    check_args_complete(args, 'train')

    args = get_args_parser_train().parse_args(['--decay-rate', '0.25'])
    assert args.gamma == 0.25
    assert not hasattr(args, 'decay_rate')
    assert scheduler_kwargs(args)['gamma'] == 0.25

    args = get_args_parser_train().parse_args(['--dr', '0.3'])
    assert args.gamma == 0.3
    assert not hasattr(args, 'decay_rate')
    assert scheduler_kwargs(args)['gamma'] == 0.3

    args = get_args_parser_train().parse_args(
        ['--decay-rate', '0.25', '--gamma', '0.4']
    )
    assert args.gamma == 0.4


def test_scheduler_kwargs_accepts_legacy_decay_rate_field():
    args = SimpleNamespace(gamma=0.8, decay_rate=0.25)
    assert scheduler_kwargs(args)['gamma'] == 0.25

    parsed_args = get_args_parser_train().parse_args([])
    vars(parsed_args)['decay_rate'] = 0.25
    check_args_complete(parsed_args, 'train')
    assert scheduler_kwargs(parsed_args)['gamma'] == 0.25


def test_parser_rejects_unknown_training_args():
    parser = get_args_parser_train()

    with pytest.raises(SystemExit):
        parser.parse_args(['--not-a-real-option'])


def test_scheduler_kwargs_defaults_match_train_parser():
    parsed_args = get_args_parser_train().parse_args([])
    partial_args = SimpleNamespace()

    parsed_defaults = scheduler_kwargs(parsed_args)
    partial_defaults = scheduler_kwargs(partial_args)

    for key in (
        'gamma',
        'min_lr',
        'step_size',
        'plateau_mode',
        'plateau_factor',
        'plateau_patience',
        'plateau_threshold',
        'plateau_threshold_mode',
        'plateau_eps',
    ):
        assert partial_defaults[key] == parsed_defaults[key]


def test_get_args_parser_rejects_unknown_script_type():
    with pytest.raises(ValueError, match='Unknown Equitrain script type'):
        get_args_parser('unknown')


def test_jax_scheduler_defaults_match_parser_monitor():
    parsed_args = get_args_parser_train().parse_args([])
    partial_controller = create_scheduler_controller(SimpleNamespace(), initial_lr=1.0)

    assert partial_controller.monitor == parsed_args.scheduler_monitor


def test_jax_plateau_scheduler_respects_eps():
    args = get_args_parser_train().parse_args([])
    args.scheduler = 'plateau'
    args.plateau_patience = 0
    args.plateau_factor = 0.999999
    args.plateau_eps = 1e-3

    controller = create_scheduler_controller(args, initial_lr=1.0)
    controller.register_initial_metric(1.0, epoch=0)

    assert not controller.update_after_epoch(metric=2.0, epoch=1)
    assert controller.current_lr == 1.0

    args.plateau_factor = 0.5
    controller = create_scheduler_controller(args, initial_lr=1.0)
    controller.register_initial_metric(1.0, epoch=0)

    assert controller.update_after_epoch(metric=2.0, epoch=1)
    assert controller.current_lr == 0.5
