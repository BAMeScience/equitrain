from types import SimpleNamespace

from equitrain.argparser import get_args_parser_train
from equitrain.backends.scheduler_common import scheduler_kwargs


def test_decay_rate_alias_overrides_gamma():
    args = get_args_parser_train().parse_args([])
    assert args.gamma == 0.8
    assert args.decay_rate is None
    assert scheduler_kwargs(args)['gamma'] == 0.8

    args = get_args_parser_train().parse_args(['--decay-rate', '0.25'])
    assert args.gamma == 0.25
    assert scheduler_kwargs(args)['gamma'] == 0.25

    args = get_args_parser_train().parse_args(['--dr', '0.3'])
    assert args.gamma == 0.3
    assert scheduler_kwargs(args)['gamma'] == 0.3

    args = get_args_parser_train().parse_args(
        ['--decay-rate', '0.25', '--gamma', '0.4']
    )
    assert args.gamma == 0.4


def test_scheduler_kwargs_accepts_legacy_decay_rate_field():
    args = SimpleNamespace(gamma=0.8, decay_rate=0.25)
    assert scheduler_kwargs(args)['gamma'] == 0.25
