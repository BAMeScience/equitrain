"""Native JAX GRACE parity, differentiation, padding, and bundle checks."""

import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp
import jraph
from ase import Atoms

from equitrain.backends.jax_wrappers.grace import GraceWrapper, build_module
from equitrain.data.backend_jax import AtomsToGraphs, graph_to_data, make_apply_fn
from equitrain.models.grace import GraceConfig, JaxGrace1L
from equitrain.models.grace.jax_model import save_bundle


@pytest.fixture(autouse=True)
def enable_double():
    previous = jax.config.x64_enabled
    jax.config.update('jax_enable_x64', True)
    yield
    jax.config.update('jax_enable_x64', previous)


def config(**kwargs):
    return GraceConfig(
        **(
            dict(
                atomic_numbers=(8, 1),
                r_max=3.0,
                lmax=2,
                max_order=4,
                n_rad_base=3,
                n_rad_max=3,
                prod_func_n_max=4,
                embedding_size=4,
                n_mlp_dens=3,
                radial_hidden=(5,),
                readout_hidden=(5,),
                avg_n_neigh=2.5,
                energy_scale=0.7,
                atomic_energies=(-1.0, -0.2),
            )
            | kwargs
        )
    )


def atoms():
    return Atoms(
        'OH2',
        positions=[[0.2, 0.3, 0.4], [1.1, 0.4, 0.6], [0.4, 1.2, 0.7]],
        cell=[[3.7, 0.2, 0.1], [0.1, 3.8, 0.2], [0.0, 0.3, 3.9]],
        pbc=True,
    )


def graph(structure=None):
    return AtomsToGraphs([8, 1], 3.0, dtype=np.float64).convert(
        atoms() if structure is None else structure
    )


def torch_pair(c):
    torch = pytest.importorskip('torch')
    from equitrain.models.grace import Grace1L

    torch.manual_seed(41)
    native = Grace1L(c).double()
    module = JaxGrace1L(c)
    return native, module, module.variables_from_torch(native)


@pytest.mark.parametrize('lmax', range(5))
@pytest.mark.parametrize('max_order', range(1, 5))
def test_torch_intermediate_features(lmax, max_order):
    import torch

    native, module, variables = torch_pair(config(lmax=lmax, max_order=max_order))
    data = graph_to_data(graph(), 2, dtype=jnp.float64)
    centers, neighbors = data['edge_index']
    vectors = data['positions'][neighbors] - data['positions'][centers] + data['shifts']
    inputs = [vectors, data['node_attrs_index'], centers, neighbors]
    expected = native.features(*[torch.from_numpy(np.array(x)) for x in inputs])
    actual = module.features(variables, *inputs)
    for name in actual:
        np.testing.assert_allclose(
            actual[name],
            expected[name].detach().numpy(),
            rtol=2e-10,
            atol=2e-11,
            err_msg=name,
        )


@pytest.mark.parametrize(
    'lmax,max_order', [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (2, 1)]
)
def test_jit_energy_force_stress_and_training_gradients(lmax, max_order):
    from types import SimpleNamespace

    import torch

    from equitrain.backends.torch_wrappers.grace import GraceWrapper as TorchWrapper
    from equitrain.data.atomic import AtomicNumberTable
    from equitrain.data.backend_torch.atoms_to_graphs import (
        AtomsToGraphs as TorchGraphs,
    )

    native, module, variables = torch_pair(config(lmax=lmax, max_order=max_order))
    wrapper = GraceWrapper(
        module=module,
        config=asdict(module.config),
        compute_force=True,
        compute_stress=True,
    )
    data = graph_to_data(graph(), 2, dtype=jnp.float64)
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch_graph = TorchGraphs(
            AtomicNumberTable([8, 1]), radius=3.0, r_edges=True
        ).convert(atoms())
    finally:
        torch.set_default_dtype(old)
    torch_wrapper = TorchWrapper(
        SimpleNamespace(forces_weight=1.0, stress_weight=1.0), native
    ).train()
    expected = torch_wrapper(torch_graph)
    actual = jax.jit(wrapper.apply)(variables, data)
    for name in actual:
        np.testing.assert_allclose(
            actual[name],
            expected[name].detach().numpy(),
            rtol=2e-8,
            atol=2e-9,
            err_msg=name,
        )
    objective = sum(value.square().sum() for value in expected.values())
    objective.backward()

    def loss(params):
        output = wrapper.apply(params, data)
        return sum(jnp.sum(value**2) for value in output.values())

    gradients = jax.jit(jax.grad(loss))(variables)['params']
    for name, parameter in native.named_parameters():
        np.testing.assert_allclose(
            gradients[name], parameter.grad.numpy(), rtol=2e-7, atol=2e-8, err_msg=name
        )


def test_padding_isolated_atoms_and_graph_pipeline_double_precision():
    c = config()
    module = JaxGrace1L(c)
    variables = module.init(jax.random.key(17), jnp.float64)
    wrapper = GraceWrapper(
        module=module, config=asdict(c), compute_force=True, compute_stress=True
    )
    apply = jax.jit(make_apply_fn(wrapper, 2))
    isolated = Atoms('H', positions=[[5.0, 0.0, 0.0]], cell=[15.0, 15.0, 15.0])
    packed = jraph.batch_np([graph(), graph(isolated)])
    expected = apply(variables, packed)
    padded = jraph.pad_with_graphs(
        packed, n_node=7, n_edge=int(packed.n_edge.sum()) + 5, n_graph=4
    )
    actual = apply(variables, padded)
    for name, count in [('energy', 2), ('forces', 4), ('stress', 2)]:
        np.testing.assert_allclose(
            actual[name][:count], expected[name], atol=1e-11, rtol=1e-10
        )
        np.testing.assert_array_equal(actual[name][count:], 0.0)
    np.testing.assert_array_equal(actual['forces'][3], 0.0)
    np.testing.assert_array_equal(actual['stress'][1], 0.0)
    assert actual['energy'].dtype == jnp.float64
    assert packed.nodes.positions.dtype == np.float64
    np.testing.assert_array_equal(packed.nodes.positions[:3], atoms().positions)
    # An unpadded single graph must retain all its atoms in node_mask.
    data = graph_to_data(graph(), 2, dtype=jnp.float64)
    assert np.all(data['node_mask'])
    assert data['positions'].dtype == jnp.float64


def test_symmetries_and_finite_difference_periodic_stress():
    c = config()
    module = JaxGrace1L(c)
    variables = module.init(jax.random.key(8), jnp.float64)
    wrapper = GraceWrapper(
        module=module, config=asdict(c), compute_force=True, compute_stress=True
    )
    apply = jax.jit(make_apply_fn(wrapper, 2))
    structure = atoms()
    expected = apply(variables, graph(structure))
    q, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))
    q[:, 0] *= -np.linalg.det(q)
    moved = structure.copy()
    moved.positions = structure.positions @ q + [0.3, -0.2, 0.8]
    moved.set_cell(structure.cell.array @ q)
    actual = apply(variables, graph(moved))
    np.testing.assert_allclose(actual['energy'], expected['energy'], atol=1e-9)
    np.testing.assert_allclose(
        actual['forces'], np.asarray(expected['forces']) @ q, atol=1e-8
    )
    np.testing.assert_allclose(
        actual['stress'], q.T @ np.asarray(expected['stress']) @ q, atol=1e-9
    )
    permutation = [2, 0, 1]
    permuted = apply(variables, graph(structure[permutation]))
    np.testing.assert_allclose(
        permuted['forces'], expected['forces'][np.array(permutation)], atol=1e-10
    )
    h = 1e-5
    for i in range(3):
        plus, minus = structure.copy(), structure.copy()
        plus.positions[0, i] += h
        minus.positions[0, i] -= h
        fd = -(
            apply(variables, graph(plus))['energy']
            - apply(variables, graph(minus))['energy']
        ) / (2 * h)
        np.testing.assert_allclose(
            expected['forces'][0, i], fd[0], rtol=2e-6, atol=1e-8
        )
    for i, j in [(0, 0), (0, 1), (1, 2)]:
        strain = np.zeros((3, 3))
        strain[i, j] += h / 2
        strain[j, i] += h / 2
        plus, minus = structure.copy(), structure.copy()
        plus.set_cell(structure.cell.array @ (np.eye(3) + strain), scale_atoms=True)
        minus.set_cell(structure.cell.array @ (np.eye(3) - strain), scale_atoms=True)
        fd = (
            apply(variables, graph(plus))['energy']
            - apply(variables, graph(minus))['energy']
        ) / (2 * h * structure.get_volume())
        np.testing.assert_allclose(
            expected['stress'][0, i, j], fd[0], rtol=2e-6, atol=1e-9
        )


@pytest.mark.parametrize(
    'force,stress', [(False, False), (True, False), (False, True), (True, True)]
)
def test_flags_float32_and_invalid_cells(force, stress):
    module = JaxGrace1L(config())
    variables = module.init(jax.random.key(2), jnp.float32)
    wrapper = GraceWrapper(
        module=module,
        config=asdict(module.config),
        compute_force=force,
        compute_stress=stress,
    )
    data = graph_to_data(graph(), 2)
    prediction = jax.jit(wrapper.apply)(variables, data)
    assert (prediction['forces'] is not None) == force
    assert (prediction['stress'] is not None) == stress
    assert prediction['energy'].dtype == jnp.float32
    assert np.isfinite(prediction['energy']).all()
    double_variables = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64), variables
    )
    double_data = graph_to_data(graph(), 2, dtype=jnp.float64)
    double_prediction = wrapper.apply(double_variables, double_data)
    for name, value in prediction.items():
        if value is not None:
            assert np.isfinite(value).all()
            np.testing.assert_allclose(
                value, double_prediction[name], rtol=3e-5, atol=2e-6
            )
    if stress:
        data = dict(data, cell=jnp.zeros((1, 3, 3)))
        with pytest.raises(ValueError, match='volume'):
            wrapper.apply(variables, data)
        assert np.isnan(jax.jit(wrapper.apply)(variables, data)['stress']).all()


def test_bundle_initialization_loading_and_no_torch_runtime(tmp_path):
    from equitrain.backends.jax_utils import load_model_bundle
    from equitrain.backends.jax_wrappers import create_wrapper

    path = save_bundle(tmp_path / 'model', config(), seed=12, dtype='float64')
    loaded = load_model_bundle(str(path), dtype='float64', wrapper='grace')
    wrapper = create_wrapper('grace', module=loaded.module, config=loaded.config)
    result = jax.jit(make_apply_fn(wrapper, 2))(loaded.params, graph())
    assert np.isfinite(result['energy']).all()
    single = make_apply_fn(wrapper, 2)(loaded.params, graph())
    np.testing.assert_allclose(result['energy'], single['energy'], atol=1e-11)
    with pytest.raises(FileExistsError):
        save_bundle(path, config())
    low = load_model_bundle(str(path), dtype='float32', wrapper='grace')
    assert low.params['params']['embedding'].dtype == jnp.float32
    code = (
        'import sys; from equitrain.models.grace import JaxGrace1L, GraceConfig; '
        'JaxGrace1L(GraceConfig((1,), lmax=1)); '
        'assert "torch" not in sys.modules; assert "tensorflow" not in sys.modules; '
        'assert "tensorpotential" not in sys.modules'
    )
    subprocess.run([sys.executable, '-c', code], check=True, timeout=60)


def test_invalid_architecture_and_compute_inputs():
    with pytest.raises(TypeError, match='zbl_cutoff'):
        build_module(asdict(config()) | {'zbl_cutoff': 2.0})
    module, template = build_module(asdict(config()))
    wrapper = GraceWrapper(module=module, config=asdict(module.config))
    with pytest.raises(ValueError, match='positions'):
        wrapper.apply(template, {})
    with pytest.raises(ValueError, match='cutoff'):
        wrapper.r_max = 5.0
    wrapper.with_compute_flags(force=True, stress=True)
    assert wrapper.compute_force and wrapper.compute_stress


def test_rejects_bad_parameter_shapes_and_preserves_converted_weights(tmp_path):
    from equitrain.backends.jax_utils import load_model_bundle

    native, module, variables = torch_pair(config())
    output = save_bundle(tmp_path / 'converted', module.config, variables=variables)
    loaded = load_model_bundle(str(output), dtype='float64', wrapper='grace')
    for name, value in variables['params'].items():
        np.testing.assert_array_equal(value, loaded.params['params'][name])
    bad = {'params': dict(variables['params'])}
    bad['params']['embedding'] = jnp.zeros((1, 1))
    with pytest.raises(ValueError, match='shape'):
        save_bundle(tmp_path / 'invalid', module.config, variables=bad)


def test_worker_converter_dtype_is_explicit_and_cache_separated(tmp_path):
    from equitrain.data.atomic import AtomicNumberTable
    from equitrain.data.backend_jax.loaders_impl import (
        StreamingDatasetSpec,
        _atoms_to_graph,
    )

    jax.config.update('jax_enable_x64', False)  # Simulate a fresh spawned worker.
    values = []
    for dtype in ('float32', 'float64'):
        spec = StreamingDatasetSpec(path=tmp_path / 'unused.h5', dtype=dtype)
        result = _atoms_to_graph(
            atoms=atoms(),
            spec=spec,
            cutoff=3.0,
            z_table=AtomicNumberTable([8, 1]),
            niggli_reduce=False,
        )
        assert result.nodes.positions.dtype == np.dtype(dtype)
        assert result.edges.shifts.dtype == np.dtype(dtype)
        values.append(result.nodes.positions)
    np.testing.assert_array_equal(values[1], atoms().positions)
    assert np.any(values[0].astype(np.float64) != values[1])
