from __future__ import annotations

import numpy as np
import pytest

from equitrain.data.format_hdf5 import HDF5Dataset
from equitrain.data.format_lmdb import convert_lmdb_to_hdf5
from equitrain.data.format_lmdb import lmdb as lmdb_module


class _DummyLmdbDataset:
    """Minimal stand-in for ``AseDBDataset``."""

    def __init__(self, records):
        self._records = records

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return iter(self._records)


@pytest.fixture
def lmdb_records():
    return [
        {
            'pos': np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=float),
            'atomic_numbers': np.array([1, 8], dtype=int),
            'cell': np.array([3.0, 3.0, 3.0], dtype=float),
            'pbc': [True, True, True],
            'energy': 1.234,
            'forces': np.zeros((2, 3), dtype=float),
            'stress': np.arange(6, dtype=float),
        },
        {
            'pos': np.array([[0.1, 0.0, 0.0]], dtype=float),
            'atomic_numbers': np.array([6], dtype=int),
            'cell': np.array(
                [2.0, 2.0, 2.0, 90.0, 90.0, 120.0], dtype=float
            ),  # lengths + angles
            'pbc': [False, False, False],
            'energy': -0.5,
            'forces': np.array([[0.1, 0.2, 0.3]], dtype=float),
            'stress': np.arange(9, dtype=float).reshape(3, 3),
        },
    ]


def test_convert_lmdb_to_hdf5(monkeypatch, tmp_path, lmdb_records):
    # Monkeypatch the dataset constructor used inside the converter.
    monkeypatch.setattr(
        lmdb_module,
        '_load_aselmdb_dataset',
        lambda: (lambda config: _DummyLmdbDataset(lmdb_records)),
    )

    dst = convert_lmdb_to_hdf5(
        src=tmp_path / 'dummy.aselmdb',
        dst=tmp_path / 'out.h5',
        overwrite=True,
        show_progress=False,
    )

    with HDF5Dataset(dst, mode='r') as dataset:
        assert len(dataset) == len(lmdb_records)

        first = dataset[0]
        assert first.get_atomic_numbers().tolist() == [1, 8]
        np.testing.assert_allclose(first.get_cell(), np.diag([3.0, 3.0, 3.0]))
        np.testing.assert_allclose(first.get_forces(), np.zeros((2, 3)))
        np.testing.assert_allclose(first.calc.get_potential_energy(), 1.234)
        np.testing.assert_allclose(first.calc.get_stress(), np.arange(6))

        second = dataset[1]
        assert second.get_atomic_numbers().tolist() == [6]
        # angles should have been expanded into a 3x3 cell matrix
        assert second.get_cell().shape == (3, 3)
        np.testing.assert_allclose(second.get_forces(), np.array([[0.1, 0.2, 0.3]]))
        # stress converted from 3x3 to Voigt notation
        np.testing.assert_allclose(
            second.calc.get_stress(), np.array([0.0, 4.0, 8.0, 5.0, 2.0, 1.0])
        )
