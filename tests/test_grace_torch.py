"""Physics and Equitrain integration checks; no external models are used."""

import json
import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torch_geometric')
from ase import Atoms
from torch_geometric.data import Batch

from equitrain.backends.torch_model import get_model
from equitrain.backends.torch_wrappers.grace import GraceWrapper
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_torch.atoms_to_graphs import AtomsToGraphs
from equitrain.models.grace import Grace1L, GraceConfig
from equitrain.models.grace.layers import chebyshev_radial, spherical_harmonics


def small_config(**kwargs):
    settings = dict(
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
    )
    return GraceConfig(**(settings | kwargs))


def molecule():
    return Atoms(
        'OH2',
        positions=[[0.2, 0.3, 0.4], [1.1, 0.4, 0.6], [0.4, 1.2, 0.7]],
        cell=[[3.7, 0.2, 0.1], [0.1, 3.8, 0.2], [0.0, 0.3, 3.9]],
        pbc=True,
    )


def graph(atoms):
    # Use double precision coordinates even if another test changes Torch's default.
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        return AtomsToGraphs(
            AtomicNumberTable([8, 1]), radius=3.0, r_edges=True
        ).convert(atoms)
    finally:
        torch.set_default_dtype(old)


def wrapper(**kwargs):
    torch.manual_seed(18)
    model = Grace1L(small_config(**kwargs)).double()
    return GraceWrapper(
        SimpleNamespace(forces_weight=1.0, stress_weight=1.0), model
    ).eval()


@pytest.mark.parametrize('lmax', range(5))
@pytest.mark.parametrize('max_order', range(1, 5))
def test_symmetries_and_batching(lmax, max_order):
    model = wrapper(lmax=lmax, max_order=max_order)
    atoms = molecule()
    original = model(graph(atoms))
    # Orthogonal matrix including reflection tests O(3), rather than SO(3) alone.
    q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(3, 3)))
    q[:, 0] *= -np.linalg.det(q)
    transformed = atoms.copy()
    transformed.positions = atoms.positions @ q + [0.8, -0.3, 0.2]
    transformed.set_cell(atoms.cell.array @ q)
    moved = model(graph(transformed))
    torch.testing.assert_close(
        moved['energy'], original['energy'], rtol=1e-9, atol=1e-10
    )
    torch.testing.assert_close(
        moved['forces'], original['forces'] @ torch.tensor(q), rtol=1e-8, atol=1e-9
    )
    expected_stress = torch.tensor(q).T @ original['stress'] @ torch.tensor(q)
    torch.testing.assert_close(moved['stress'], expected_stress, rtol=1e-8, atol=1e-9)
    permutation = [2, 0, 1]
    permuted = model(graph(atoms[permutation]))
    torch.testing.assert_close(permuted['energy'], original['energy'])
    torch.testing.assert_close(permuted['forces'], original['forces'][permutation])
    batch = model(Batch.from_data_list([graph(atoms), graph(transformed)]))
    torch.testing.assert_close(
        batch['energy'], torch.cat([original['energy'], moved['energy']])
    )
    torch.testing.assert_close(
        batch['forces'], torch.cat([original['forces'], moved['forces']])
    )
    torch.testing.assert_close(
        batch['stress'], torch.cat([original['stress'], moved['stress']])
    )
    torch.testing.assert_close(
        original['forces'].sum(0),
        torch.zeros(3, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )


def test_force_and_periodic_stress_finite_differences():
    model = wrapper()
    atoms = molecule()
    result = model(graph(atoms))
    h = 1e-5
    for atom in range(len(atoms)):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[atom, axis] += h
            minus.positions[atom, axis] -= h
            fd = -(model(graph(plus))['energy'] - model(graph(minus))['energy']) / (
                2 * h
            )
            torch.testing.assert_close(
                result['forces'][atom, axis], fd.squeeze(), rtol=2e-6, atol=1e-8
            )
    for i in range(3):
        for j in range(i, 3):
            strain = np.zeros((3, 3))
            strain[i, j] += 0.5 * h
            strain[j, i] += 0.5 * h
            plus, minus = atoms.copy(), atoms.copy()
            plus.set_cell(atoms.cell.array @ (np.eye(3) + strain), scale_atoms=True)
            minus.set_cell(atoms.cell.array @ (np.eye(3) - strain), scale_atoms=True)
            fd = (model(graph(plus))['energy'] - model(graph(minus))['energy']) / (
                2 * h * atoms.get_volume()
            )
            torch.testing.assert_close(
                result['stress'][0, i, j], fd.squeeze(), rtol=2e-6, atol=1e-9
            )


@pytest.mark.parametrize(
    'force,stress', [(False, False), (True, False), (False, True), (True, True)]
)
def test_derivative_modes_and_no_grad(force, stress):
    model = wrapper()
    model.compute_force, model.compute_stress = force, stress
    with torch.no_grad():
        prediction = model(graph(molecule()))
    assert (prediction['forces'] is not None) == force
    assert (prediction['stress'] is not None) == stress
    assert prediction['energy'].shape == (1,)


def test_force_and_stress_training_updates_and_reload(tmp_path):
    model = wrapper().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = graph(molecule())
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    prediction = model(data)
    # Derivative-only loss requires mixed second derivatives with model weights.
    loss = prediction['forces'].square().sum() + prediction['stress'].square().sum()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert any(not torch.equal(before[k], p) for k, p in model.named_parameters())
    config_path = tmp_path / 'model.json'
    model.model.save_config(config_path)
    args = SimpleNamespace(
        model=str(config_path),
        model_wrapper='grace',
        dtype='float64',
        forces_weight=1.0,
        stress_weight=1.0,
    )
    reloaded = get_model(args).eval()
    reloaded.load_state_dict(model.state_dict())
    model.eval()
    for name, value in model(data).items():
        torch.testing.assert_close(value, reloaded(data)[name])
    path = tmp_path / 'model.pt'
    torch.save(model.model, path)
    args.model = str(path)
    pickled = get_model(args).eval()
    torch.testing.assert_close(model(data)['energy'], pickled(data)['energy'])


def test_isolated_atoms_empty_edges_and_cell_errors():
    model = wrapper().train()
    atoms = Atoms('OH', positions=[[0, 0, 0], [5, 0, 0]], cell=[15, 15, 15])
    prediction = model(graph(atoms))
    assert torch.count_nonzero(prediction['forces']) == 0
    assert torch.count_nonzero(prediction['stress']) == 0
    (
        prediction['forces'].square().sum() + prediction['stress'].square().sum()
    ).backward()
    atoms.set_cell([0, 0, 0])
    with pytest.raises(ValueError, match='volume'):
        model(graph(atoms))
    model.compute_stress = False
    assert torch.isfinite(model(graph(atoms))['energy']).all()


def test_cutoff_smoothness_and_axis_second_derivatives():
    r = torch.tensor(
        [3.0 - 1e-5, 3.0, 3.0 + 1e-5], dtype=torch.float64, requires_grad=True
    )
    basis = chebyshev_radial(r, 3.0, 4, 5)
    first = torch.autograd.grad(basis.sum(), r, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), r)[0]
    assert torch.count_nonzero(basis[1:]) == 0
    torch.testing.assert_close(
        first[1:], torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        second[1:], torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0
    )
    vectors = torch.eye(3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: spherical_harmonics(x / x.norm(dim=-1, keepdim=True), 4), (vectors,)
    )
    assert torch.autograd.gradgradcheck(
        lambda x: spherical_harmonics(x / x.norm(dim=-1, keepdim=True), 4), (vectors,)
    )


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(lmax=5),
        dict(max_order=0),
        dict(r_max=0),
        dict(avg_n_neigh=0),
        dict(atomic_numbers=(1, 1)),
        dict(n_rad_max=0),
        dict(atomic_energies=(0.0,)),
        dict(cutoff_order=2),
        dict(energy_scale=float('nan')),
    ],
)
def test_invalid_configs(kwargs):
    with pytest.raises(ValueError):
        small_config(**kwargs)


def test_unsupported_element_and_cutoff_override():
    model = wrapper()
    data = graph(molecule())
    data.atomic_numbers[0] = 6
    with pytest.raises(ValueError, match='element'):
        model(data)
    with pytest.raises(ValueError, match='cutoff'):
        model.r_max = 4.0


def test_float32_and_no_tensorflow_runtime():
    import copy

    reference = wrapper()
    expected = reference(graph(molecule()))
    model = copy.deepcopy(reference).float()
    result = model(graph(molecule()))
    assert all(
        value.dtype == torch.float32 and torch.isfinite(value).all()
        for value in result.values()
    )
    for name in result:
        torch.testing.assert_close(
            result[name].double(), expected[name], rtol=3e-5, atol=2e-6
        )
    # A separate interpreter proves that the Torch model never imports TensorFlow.
    code = (
        'import sys; from equitrain.models.grace import Grace1L, GraceConfig; '
        'Grace1L(GraceConfig((1,), lmax=1)); '
        'assert "tensorflow" not in sys.modules; assert "tensorpotential" not in sys.modules'
    )
    subprocess.run([sys.executable, '-c', code], check=True, timeout=60)


def test_json_unknown_options_fail(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(asdict(small_config()) | {'zbl_cutoff': 2.0}))
    with pytest.raises(TypeError, match='zbl_cutoff'):
        Grace1L.from_json(path)
