"""
Helpers for constructing lightweight MACE models used in tests.

The real MACE dependency tree is optional. All imports are done lazily so that
test modules that do not need MACE can still be collected/executed even when
the package is not installed (or built against a different torch version).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk

_MACE_MODULES: SimpleNamespace | None = None
_MACE_IMPORT_ERROR: Exception | None = None


def _load_mace_modules() -> SimpleNamespace | None:
    """
    Attempt to import the optional MACE stack.
    Returns a namespace with the required symbols or None when unavailable.
    """
    global _MACE_MODULES, _MACE_IMPORT_ERROR
    if _MACE_MODULES is not None or _MACE_IMPORT_ERROR is not None:
        return _MACE_MODULES

    try:  # pragma: no cover - optional dependency guard
        from mace.data.atomic_data import AtomicData
        from mace.data.utils import config_from_atoms
        from mace.tools import build_default_arg_parser, check_args, torch_geometric
        from mace.tools.model_script_utils import (
            configure_model as configure_model_torch,
        )
        from mace.tools.multihead_tools import AtomicNumberTable, prepare_default_head
        from mace.tools.torch_geometric.batch import Batch
    except Exception as exc:  # pragma: no cover
        _MACE_IMPORT_ERROR = exc
        return None

    _MACE_MODULES = SimpleNamespace(
        AtomicData=AtomicData,
        config_from_atoms=config_from_atoms,
        torch_geometric=torch_geometric,
        configure_model_torch=configure_model_torch,
        AtomicNumberTable=AtomicNumberTable,
        prepare_default_head=prepare_default_head,
        Batch=Batch,
        build_default_arg_parser=build_default_arg_parser,
        check_args=check_args,
    )
    return _MACE_MODULES


def _require_mace():  # pragma: no cover - helper for skipping tests
    modules = _load_mace_modules()
    if modules is None:
        import pytest

        pytest.skip(f'MACE is unavailable: {_MACE_IMPORT_ERROR!r}')
    return modules


def _build_structures() -> list[Atoms]:
    structures: list[Atoms] = []

    base = bulk('NaCl', 'rocksalt', a=5.0).repeat((1, 1, 1))
    structures.append(base)

    displaced = base.copy()
    displaced.positions += 0.05 * np.random.default_rng(seed=0).normal(
        size=displaced.positions.shape
    )
    structures.append(displaced)

    return structures


def _build_statistics(zs: Iterable[int]):
    modules = _require_mace()
    AtomicNumberTable = modules.AtomicNumberTable
    return {
        'mean': [0.0],
        'std': [1.0],
        'avg_num_neighbors': 4.0,
        'r_max': 3.5,
        'atomic_numbers': AtomicNumberTable(sorted(set(int(z) for z in zs))),
        'atomic_energies': [0.0 for _ in zs],
    }


def _create_args(statistics: dict):
    modules = _require_mace()
    build_default_arg_parser = modules.build_default_arg_parser
    check_args = modules.check_args
    prepare_default_head = modules.prepare_default_head

    args_list = [
        '--name',
        'equitrain-test',
        '--interaction_first',
        'RealAgnosticInteractionBlock',
        '--interaction',
        'RealAgnosticResidualInteractionBlock',
        '--num_channels',
        '4',
        '--max_L',
        '1',
        '--max_ell',
        '1',
        '--num_interactions',
        '1',
        '--correlation',
        '1',
        '--num_radial_basis',
        '4',
        '--num_cutoff_basis',
        '4',
        '--MLP_irreps',
        '4x0e',
        '--distance_transform',
        'Agnesi',
        '--pair_repulsion',
    ]

    args = build_default_arg_parser().parse_args(args_list)
    args, _ = check_args(args)

    args.mean = statistics['mean']
    args.std = statistics['std']
    args.compute_energy = True
    args.compute_forces = False
    args.compute_dipole = False
    args.compute_polarizability = False
    args.compute_stress = True
    args.loss = 'energy'
    args.device = 'cpu'
    args.train_file = ''
    args.valid_file = ''
    args.test_file = ''
    args.test_dir = ''
    args.E0s = None
    args.statistics_file = None
    args.key_specification = None
    args.valid_fraction = None
    args.config_type_weights = None
    args.keep_isolated_atoms = False
    args.only_cueq = False
    args.apply_cutoff = True
    args.use_reduced_cg = False
    args.use_so3 = False
    args.embedding_specs = None
    args.use_embedding_readout = False
    args.use_last_readout_only = False
    args.use_agnostic_product = False
    args.heads = prepare_default_head(args)
    args.avg_num_neighbors = statistics['avg_num_neighbors']
    args.r_max = statistics['r_max']

    return args


def _write_small_mace_model(path: Path) -> None:
    modules = _require_mace()
    AtomicData = modules.AtomicData
    config_from_atoms = modules.config_from_atoms
    torch_geometric = modules.torch_geometric
    configure_model_torch = modules.configure_model_torch
    Batch = modules.Batch

    structures = _build_structures()
    zs = [int(z) for atoms in structures for z in atoms.get_atomic_numbers()]
    statistics = _build_statistics(zs)

    atomic_data_list = []
    for atoms in structures:
        config = config_from_atoms(atoms)
        config.pbc = [bool(x) for x in config.pbc]
        atomic_data_list.append(
            AtomicData.from_config(
                config,
                z_table=statistics['atomic_numbers'],
                cutoff=float(statistics['r_max']),
            )
        )

    batch = torch_geometric.batch.Batch.from_data_list(atomic_data_list)
    batch = batch.to(torch.float32)
    Batch.validate(batch)

    args = _create_args(statistics)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'.*TorchScript type system doesn\'t support instance-level annotations.*',
            category=UserWarning,
        )
        torch_model, _ = configure_model_torch(
            args,
            train_loader=[],
            atomic_energies=statistics['atomic_energies'],
            heads=args.heads,
            z_table=statistics['atomic_numbers'],
        )
    torch_model = torch_model.float().eval()

    torch.save(torch_model, path)


def get_mace_model_path() -> Path:
    """
    Ensure the tiny MACE model used by the tests exists and return its path.
    Will skip the caller if MACE cannot be imported.
    """
    path = Path(__file__).resolve().parents[2] / 'tests' / 'mace.model'
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        _write_small_mace_model(path)
    return path


__all__ = ['get_mace_model_path']
