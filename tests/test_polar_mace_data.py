from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.configuration import Configuration
from equitrain.data.format_hdf5 import HDF5Dataset


def _atoms(
    *,
    charge=-1.0,
    spin=2.0,
    external_field=(0.1, -0.2, 0.3),
    source_id=1,
    reaction_id=7,
    state_id=1,
):
    atoms = Atoms(
        symbols='OH',
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]], dtype=float),
        cell=np.eye(3) * 5.0,
        pbc=[False, False, False],
    )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.25,
        forces=np.zeros((len(atoms), 3), dtype=float),
        stress=np.zeros((3, 3), dtype=float),
    )
    atoms.info['virials'] = np.zeros((3, 3), dtype=float)
    atoms.info['dipole'] = np.array([1.0, 2.0, 3.0], dtype=float)
    atoms.info['charge'] = charge
    atoms.info['spin'] = spin
    atoms.info['external_field'] = np.asarray(external_field, dtype=float)
    atoms.info['source_id'] = source_id
    atoms.info['reaction_id'] = reaction_id
    atoms.info['state_id'] = state_id
    atoms.info['energy_weight'] = 1.0
    atoms.info['forces_weight'] = 1.0
    atoms.info['stress_weight'] = 1.0
    atoms.info['virials_weight'] = 0.0
    atoms.info['dipole_weight'] = 0.0
    return atoms


def test_configuration_preserves_polar_mace_metadata():
    atoms = _atoms()
    atoms.info['net_charge'] = -2.0
    atoms.info['multiplicity'] = 3.0
    atoms.info['field'] = [0.4, 0.5, 0.6]

    config = Configuration.from_atoms(
        atoms,
        total_charge_key='net_charge',
        total_spin_key='multiplicity',
        external_field_key='field',
    )

    assert config.total_charge == -2.0
    assert config.total_spin == 3.0
    np.testing.assert_allclose(config.external_field, [0.4, 0.5, 0.6])
    assert config.source_id == 1
    assert config.reaction_id == 7
    assert config.state_id == 1

    roundtrip = config.to_atoms()
    assert roundtrip.info['charge'] == -2.0
    assert roundtrip.info['total_charge'] == -2.0
    assert roundtrip.info['spin'] == 3.0
    assert roundtrip.info['total_spin'] == 3.0
    np.testing.assert_allclose(roundtrip.info['external_field'], [0.4, 0.5, 0.6])
    assert roundtrip.info['source_id'] == 1
    assert roundtrip.info['reaction_id'] == 7
    assert roundtrip.info['state_id'] == 1


def test_hdf5_roundtrip_stores_polar_mace_metadata(tmp_path):
    path = tmp_path / 'polar.h5'
    with HDF5Dataset(path, mode='w') as dataset:
        dataset[0] = _atoms(charge=-1.5, spin=4.0, external_field=[0.2, 0.0, -0.1])

    with HDF5Dataset(path, mode='r') as dataset:
        names = dataset.file[dataset.STRUCTURES_DATASET].dtype.names
        assert 'total_charge' in names
        assert 'total_spin' in names
        assert 'external_field' in names
        assert 'source_id' in names
        assert 'reaction_id' in names
        assert 'state_id' in names

        atoms = dataset[0]
        assert atoms.info['charge'] == -1.5
        assert atoms.info['total_charge'] == -1.5
        assert atoms.info['spin'] == 4.0
        assert atoms.info['total_spin'] == 4.0
        np.testing.assert_allclose(atoms.info['external_field'], [0.2, 0.0, -0.1])
        assert atoms.info['source_id'] == 1
        assert atoms.info['reaction_id'] == 7
        assert atoms.info['state_id'] == 1
        assert dataset.reaction_metadata() == [(1, 7, 1)]


def test_hdf5_old_schema_defaults_to_neutral_singlet_zero_field(tmp_path):
    path = tmp_path / 'old.h5'
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    forces = np.zeros((1, 3), dtype=np.float64)
    atomic_numbers = np.array([1], dtype=np.int32)
    old_dtype = np.dtype(
        [
            ('offset', np.int64),
            ('length', np.int32),
            ('cell', np.float64, (3, 3)),
            ('pbc', np.bool_, (3,)),
            ('energy', np.float64),
            ('stress', np.float64, (6,)),
            ('virials', np.float64, (3, 3)),
            ('dipole', np.float64, (3,)),
            ('energy_weight', np.float32),
            ('forces_weight', np.float32),
            ('stress_weight', np.float32),
            ('virials_weight', np.float32),
            ('dipole_weight', np.float32),
        ]
    )
    entry = np.array(
        [
            (
                0,
                1,
                np.eye(3),
                np.array([False, False, False]),
                0.0,
                np.zeros(6),
                np.zeros((3, 3)),
                np.zeros(3),
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
            )
        ],
        dtype=old_dtype,
    )

    with h5py.File(path, 'w') as handle:
        magic = handle.create_group('MAGIC')
        magic['MAGIC_STRING'] = HDF5Dataset.MAGIC_STRING
        handle.create_dataset(
            HDF5Dataset.STRUCTURES_DATASET,
            data=entry,
            maxshape=(None,),
            chunks=True,
        )
        handle.create_dataset(
            HDF5Dataset.POSITIONS_DATASET,
            data=positions,
            maxshape=(None, 3),
            chunks=(HDF5Dataset._DEFAULT_CHUNK_ATOMS, 3),
        )
        handle.create_dataset(
            HDF5Dataset.FORCES_DATASET,
            data=forces,
            maxshape=(None, 3),
            chunks=(HDF5Dataset._DEFAULT_CHUNK_ATOMS, 3),
        )
        handle.create_dataset(
            HDF5Dataset.ATOMIC_NUMBERS_DATASET,
            data=atomic_numbers,
            maxshape=(None,),
            chunks=(HDF5Dataset._DEFAULT_CHUNK_ATOMS,),
        )

    with HDF5Dataset(path, mode='r') as dataset:
        atoms = dataset[0]

    assert atoms.info['charge'] == 0.0
    assert atoms.info['total_charge'] == 0.0
    assert atoms.info['spin'] == 1.0
    assert atoms.info['total_spin'] == 1.0
    np.testing.assert_allclose(atoms.info['external_field'], np.zeros(3))
    assert atoms.info['source_id'] == 0
    assert atoms.info['reaction_id'] == -1
    assert atoms.info['state_id'] == -1


def test_torch_graph_contains_polar_mace_inputs():
    pytest.importorskip('torch_geometric')
    from torch_geometric.data import Batch

    from equitrain.data.backend_torch import AtomsToGraphs

    converter = AtomsToGraphs(
        AtomicNumberTable([1, 8]),
        radius=3.0,
        r_edges=True,
        r_energy=True,
        r_forces=True,
        r_stress=True,
        r_pbc=True,
    )
    graph = converter.convert(_atoms(charge=-1.0, spin=2.0))

    assert graph.total_charge.shape == torch.Size([])
    assert graph.total_spin.shape == torch.Size([])
    assert graph.external_field.shape == (1, 3)
    assert graph.source_id.shape == torch.Size([])
    assert graph.reaction_id.shape == torch.Size([])
    assert graph.state_id.shape == torch.Size([])
    assert graph.fermi_level.shape == torch.Size([])
    assert graph.volume.shape == torch.Size([])
    assert graph.rcell.shape == (3, 3)
    assert graph.total_charge.item() == -1.0
    assert graph.total_spin.item() == 2.0
    assert graph.source_id.item() == 1
    assert graph.reaction_id.item() == 7
    assert graph.state_id.item() == 1
    np.testing.assert_allclose(graph.external_field.numpy(), [[0.1, -0.2, 0.3]])

    cell = np.eye(3) * 5.0
    expected_rcell = 2.0 * np.pi * np.linalg.inv(cell.T)
    np.testing.assert_allclose(graph.rcell.numpy(), expected_rcell)

    batch = Batch.from_data_list(
        [
            graph,
            converter.convert(_atoms(charge=0.5, spin=1.5, external_field=[0, 0, 1])),
        ]
    )
    assert batch.total_charge.shape == (2,)
    assert batch.total_spin.shape == (2,)
    assert batch.external_field.shape == (2, 3)
    assert batch.source_id.shape == (2,)
    assert batch.reaction_id.shape == (2,)
    assert batch.state_id.shape == (2,)
    assert batch.fermi_level.shape == (2,)
    assert batch.volume.shape == (2,)
    assert batch.rcell.shape == (6, 3)
    np.testing.assert_allclose(batch.total_charge.numpy(), [-1.0, 0.5])
    np.testing.assert_allclose(batch.total_spin.numpy(), [2.0, 1.5])
    np.testing.assert_array_equal(batch.source_id.numpy(), [1, 1])
    np.testing.assert_array_equal(batch.reaction_id.numpy(), [7, 7])
    np.testing.assert_array_equal(batch.state_id.numpy(), [1, 1])
    np.testing.assert_allclose(batch.external_field.numpy()[1], [0.0, 0.0, 1.0])
