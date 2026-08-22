from types import SimpleNamespace

from equitrain.argparser import check_args_complete, get_args_parser_train
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
