import pytest
import torch

pytest.importorskip('mace', reason='MACE is required for MACE integration tests.')

from equitrain import get_args_parser_train
from equitrain.utility_test import MaceWrapper
from equitrain.utility_test.mace_support import get_mace_model_path


def _make_mace_wrapper():
    args = get_args_parser_train().parse_args([])
    return MaceWrapper(args, filename_model=get_mace_model_path())


def test_same_r_max_assignment_keeps_radial_embedding():
    wrapper = _make_mace_wrapper()
    radial_embedding = wrapper.model.radial_embedding
    original_weights = torch.nn.Parameter(
        torch.full_like(radial_embedding.bessel_fn.bessel_weights, 3.0)
    )
    radial_embedding.bessel_fn.bessel_weights = original_weights

    wrapper.r_max = wrapper.r_max

    assert wrapper.model.radial_embedding is radial_embedding
    assert wrapper.model.radial_embedding.bessel_fn.bessel_weights is original_weights
    assert torch.allclose(
        wrapper.model.radial_embedding.bessel_fn.bessel_weights,
        torch.full_like(original_weights, 3.0),
    )


def test_r_max_change_supports_missing_distance_transform():
    wrapper = _make_mace_wrapper()
    radial_embedding = wrapper.model.radial_embedding
    if hasattr(radial_embedding, 'distance_transform'):
        delattr(radial_embedding, 'distance_transform')
    radial_embedding.apply_cutoff = False

    target_r_max = wrapper.r_max + 0.5
    wrapper.r_max = target_r_max

    assert wrapper.r_max == pytest.approx(target_r_max)
    assert wrapper.model.radial_embedding.cutoff_fn.r_max.item() == pytest.approx(
        target_r_max
    )
    assert not hasattr(wrapper.model.radial_embedding, 'distance_transform')
    assert wrapper.model.radial_embedding.apply_cutoff is False
