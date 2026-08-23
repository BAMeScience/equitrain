from types import SimpleNamespace

import pytest

from equitrain.argparser import get_args_parser_train


def test_torch_loss_defaults_match_train_parser():
    pytest.importorskip('torch')
    from equitrain.backends.torch_loss_fn import LossFn

    args = get_args_parser_train().parse_args([])
    loss_fn = LossFn(loss_type=args.loss_type, huber_delta=args.huber_delta)

    assert loss_fn.energy_weight == args.energy_weight
    assert loss_fn.forces_weight == args.forces_weight
    assert loss_fn.stress_weight == args.stress_weight


def test_jax_loss_defaults_match_train_parser():
    pytest.importorskip('jax')
    pytest.importorskip('jraph')
    from equitrain.backends.jax_loss_fn import LossSettings

    args = get_args_parser_train().parse_args([])
    settings = LossSettings.from_args(SimpleNamespace())

    assert settings.energy_weight == args.energy_weight
    assert settings.forces_weight == args.forces_weight
    assert settings.stress_weight == args.stress_weight
    assert settings.loss_type == args.loss_type
    assert settings.smooth_l1_beta == args.smooth_l1_beta
    assert settings.huber_delta == args.huber_delta


def test_jax_optimizer_defaults_match_train_parser():
    pytest.importorskip('optax')
    from equitrain.backends.jax_optimizer import optimizer_kwargs

    args = get_args_parser_train().parse_args([])
    kwargs = optimizer_kwargs(SimpleNamespace())

    assert kwargs['optimizer_name'] == args.opt
    assert kwargs['learning_rate'] == args.lr
    assert kwargs['weight_decay'] == 0.0
    assert kwargs['momentum'] == args.momentum
    assert kwargs['alpha'] == args.alpha
