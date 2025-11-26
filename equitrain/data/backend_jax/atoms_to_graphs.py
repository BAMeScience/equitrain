from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import jraph
from ase import Atoms

from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.configuration import Configuration, niggli_reduce_inplace
from equitrain.data.format_hdf5.dataset import HDF5Dataset

from .atoms_to_graphs_impl import (
    graph_from_configuration,
    graph_to_data,
    make_apply_fn,
)


class AtomsToGraphs:
    """Convert ASE ``Atoms`` objects into ``jraph.GraphsTuple`` objects."""

    def __init__(
        self,
        atomic_numbers: AtomicNumberTable | Sequence[int],
        r_max: float,
        *,
        niggli_reduce: bool = False,
    ) -> None:
        if atomic_numbers is None:
            raise ValueError('An atomic number table is required to build graphs.')
        if not hasattr(atomic_numbers, 'z_to_index'):
            atomic_numbers = AtomicNumberTable(list(atomic_numbers))
        self._z_table = atomic_numbers
        cutoff = float(r_max or 0.0)
        if cutoff <= 0.0:
            raise ValueError('A positive cutoff radius is required to build graphs.')
        self._cutoff = cutoff
        self._niggli = bool(niggli_reduce)

    def _to_configuration(self, atoms: Atoms) -> Configuration:
        if self._niggli:
            atoms = atoms.copy()
            niggli_reduce_inplace(atoms)
        return Configuration.from_atoms(atoms)

    def convert(self, atoms: Atoms | Configuration) -> jraph.GraphsTuple:
        config = (
            atoms if isinstance(atoms, Configuration) else self._to_configuration(atoms)
        )
        return graph_from_configuration(
            config,
            cutoff=self._cutoff,
            z_table=self._z_table,
        )

    def convert_dataset(self, dataset: HDF5Dataset) -> Iterable[jraph.GraphsTuple]:
        for idx in range(len(dataset)):
            atoms = dataset[idx]
            yield self.convert(atoms)

    def convert_file(self, data_path: Path | str) -> list[jraph.GraphsTuple]:
        if data_path is None:
            return []
        dataset = HDF5Dataset(data_path, mode='r')
        graphs: list[jraph.GraphsTuple] = []
        try:
            graphs.extend(self.convert_dataset(dataset))
        finally:
            dataset.close()
        return graphs


__all__ = [
    'AtomsToGraphs',
    'graph_from_configuration',
    'graph_to_data',
    'make_apply_fn',
]
