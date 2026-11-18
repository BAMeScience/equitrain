"""Exercise native GRACE through Equitrain's real training/export pipeline."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('accelerate')
pytest.importorskip('torch_geometric')
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from safetensors.torch import load_file

from equitrain import get_args_parser_export, get_args_parser_train, train
from equitrain.backends.torch_model import get_model
from equitrain.calculators.torch_wrapper import TorchWrapperPredictor
from equitrain.data.format_hdf5.dataset import HDF5Dataset
from equitrain.models.grace import Grace1L, GraceConfig
from equitrain.scripts.equitrain_export import export


def test_train_checkpoint_export_predict(tmp_path, monkeypatch):
    monkeypatch.setenv('ACCELERATE_USE_CPU', 'true')
    data_path = tmp_path / 'structures.h5'
    atoms = Atoms(
        'OH2',
        positions=[[0, 0, 0], [1, 0.2, 0], [0.2, 1, 0.3]],
        cell=[4, 4, 4],
        pbc=True,
    )
    with HDF5Dataset(data_path, mode='w') as dataset:
        for i in range(2):
            structure = atoms.copy()
            structure.positions[1, 0] += 0.05 * i
            structure.calc = SinglePointCalculator(
                structure,
                energy=-1.0 + 0.1 * i,
                forces=np.zeros((3, 3)),
                stress=np.zeros(6),
            )
            dataset[i] = structure
    config_path = tmp_path / 'grace.json'
    Grace1L(
        GraceConfig(
            (8, 1),
            r_max=3.0,
            lmax=2,
            n_rad_base=3,
            n_rad_max=3,
            prod_func_n_max=4,
            embedding_size=4,
            n_mlp_dens=3,
            radial_hidden=(5,),
            readout_hidden=(5,),
        )
    ).save_config(config_path)
    args = get_args_parser_train().parse_args([])
    args.model = str(config_path)
    args.model_wrapper = 'grace'
    args.train_file = args.valid_file = str(data_path)
    args.output_dir = str(tmp_path / 'training')
    args.dtype = 'float64'
    args.epochs = 1
    args.batch_size = 2
    args.lr = 1e-3
    args.train_max_steps = args.valid_max_steps = 1
    args.energy_weight = args.forces_weight = args.stress_weight = 1.0
    args.num_workers = 0
    args.pin_memory = False
    args.verbose = 0
    args.tqdm = False
    old_dtype = torch.get_default_dtype()
    try:
        train(args)
    finally:
        torch.set_default_dtype(old_dtype)
    (checkpoint,) = Path(args.output_dir).glob('best_val_epochs@*')
    weights = load_file(checkpoint / 'model.safetensors')
    assert weights
    model = get_model(
        SimpleNamespace(
            model=str(config_path),
            model_wrapper='grace',
            dtype='float64',
            forces_weight=1.0,
            stress_weight=1.0,
        )
    )
    model.load_state_dict(weights)
    assert any(
        torch.count_nonzero(p)
        for name, p in model.named_parameters()
        if 'trainable_shift' in name
    )
    export_args = get_args_parser_export().parse_args([])
    export_args.model = str(config_path)
    export_args.model_wrapper = 'grace'
    export_args.output_dir = args.output_dir
    export_args.load_checkpoint = str(checkpoint)
    export_args.model_export = str(tmp_path / 'exported.pt')
    export(export_args)
    assert Path(export_args.model_export).exists()
    # Load through the shared prediction adapter, as an end user would.
    predictor = TorchWrapperPredictor(
        model=export_args.model_export,
        model_wrapper='grace',
        device='cpu',
        default_dtype='float64',
        require_forces=True,
    )
    energies, forces = predictor.predict([atoms], require_forces=True)
    assert len(energies) == len(forces) == 1
    assert np.isfinite(energies).all() and np.isfinite(forces).all()
