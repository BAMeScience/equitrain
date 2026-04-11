from __future__ import annotations

from types import SimpleNamespace

from equitrain.backends import jax_wrappers


def test_infer_wrapper_name_prefers_explicit_argument():
    config = {'model_wrapper': 'mace'}

    assert jax_wrappers.infer_wrapper_name(config, 'ani') == 'ani'


def test_infer_wrapper_name_falls_back_to_config():
    config = {'wrapper_name': 'ani'}

    assert jax_wrappers.infer_wrapper_name(config, None) == 'ani'


def test_create_wrapper_dispatches_to_selected_module(monkeypatch):
    captured: dict[str, object] = {}

    class FakeAniWrapper:
        def __init__(self, *, module, config, compute_force, compute_stress):
            captured['module'] = module
            captured['config'] = config
            captured['compute_force'] = compute_force
            captured['compute_stress'] = compute_stress

    def fake_import_module(name: str, package: str):
        assert name == '.ani'
        assert package == 'equitrain.backends.jax_wrappers'
        return SimpleNamespace(AniWrapper=FakeAniWrapper)

    monkeypatch.setattr(jax_wrappers, 'import_module', fake_import_module)

    wrapper = jax_wrappers.create_wrapper(
        'ani',
        module='module-object',
        config={'model_wrapper': 'ani'},
        compute_force=True,
        compute_stress=False,
    )

    assert isinstance(wrapper, FakeAniWrapper)
    assert captured == {
        'module': 'module-object',
        'config': {'model_wrapper': 'ani'},
        'compute_force': True,
        'compute_stress': False,
    }


def test_get_wrapper_builder_uses_normalized_name(monkeypatch):
    def fake_import_module(name: str, package: str):
        assert name == '.ani'
        assert package == 'equitrain.backends.jax_wrappers'
        return SimpleNamespace(build_module='builder')

    monkeypatch.setattr(jax_wrappers, 'import_module', fake_import_module)

    assert jax_wrappers.get_wrapper_builder('ANI') == 'builder'
