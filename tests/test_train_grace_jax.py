"""Train, checkpoint, resume, and predict with native JAX GRACE and local data."""

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip('jax')
pytest.importorskip('optax')
pytest.importorskip('flax')

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from flax import serialization

from equitrain import get_args_parser_train, train
from equitrain.backends.jax_utils import load_model_bundle
from equitrain.calculators.jax_wrapper import JaxWrapperPredictor
from equitrain.data.format_hdf5.dataset import HDF5Dataset
from equitrain.models.grace import GraceConfig
from equitrain.models.grace.jax_model import save_bundle


def test_jax_train_resume_and_predict(tmp_path):
    previous = jax.config.x64_enabled
    try:
        _run_training(tmp_path)
    finally:
        jax.config.update('jax_enable_x64', previous)


def _run_training(tmp_path):
    structure = Atoms(
        'OH2',
        positions=[[0, 0, 0], [1, 0.2, 0], [0.2, 1, 0.3]],
        cell=[4, 4, 4],
        pbc=True,
    )
    data_path = tmp_path / 'structures.h5'
    with HDF5Dataset(data_path, mode='w') as dataset:
        for i in range(2):
            sample = structure.copy()
            sample.positions[1, 0] += 0.05 * i
            sample.calc = SinglePointCalculator(
                sample,
                energy=-1.0 + 0.1 * i,
                forces=np.zeros((3, 3)),
                stress=np.zeros(6),
            )
            dataset[i] = sample
    c = GraceConfig(
        (8, 1),
        r_max=3.0,
        lmax=1,
        n_rad_base=3,
        n_rad_max=2,
        prod_func_n_max=2,
        embedding_size=3,
        n_mlp_dens=2,
        radial_hidden=(4,),
        readout_hidden=(4,),
    )
    bundle_path = save_bundle(tmp_path / 'initial', c, seed=42)
    initial = load_model_bundle(str(bundle_path), dtype='float64', wrapper='grace')

    def args_for(epochs):
        args = get_args_parser_train().parse_args([])
        args.backend = 'jax'
        args.model_wrapper = 'grace'
        args.model = str(bundle_path)
        args.train_file = args.valid_file = str(data_path)
        args.output_dir = str(tmp_path / 'training')
        args.dtype = 'float64'
        args.jax_platform = 'cpu'
        args.epochs = epochs
        args.train_max_steps = args.valid_max_steps = 1
        args.batch_max_edges = 64
        args.batch_max_nodes = 16
        args.energy_weight = args.forces_weight = args.stress_weight = 1.0
        args.lr = 1e-3
        args.num_workers = 0
        args.prefetch_batches = 1
        args.pin_memory = False
        args.verbose = 0
        args.tqdm = False
        return args

    args = args_for(1)
    train(args)
    (checkpoint,) = Path(args.output_dir).glob('best_val_epochs@*')
    loaded = load_model_bundle(str(checkpoint), dtype='float64', wrapper='grace')
    assert (checkpoint / 'opt_state.msgpack').exists()
    assert any(
        not np.array_equal(initial.params['params'][name], value)
        for name, value in loaded.params['params'].items()
    )
    predictor = JaxWrapperPredictor(
        model=str(checkpoint),
        model_wrapper='grace',
        default_dtype='float64',
        device='cpu',
        require_forces=True,
    )
    energies, forces = predictor.predict([structure], require_forces=True)
    assert len(energies) == len(forces) == 1
    assert np.isfinite(energies).all() and np.isfinite(forces).all()
    # Resume optimizer/model state for one more epoch through the real backend.
    resumed = args_for(2)
    resumed.load_checkpoint = str(checkpoint)
    train(resumed)
    final_path = Path(args.output_dir) / 'jax_params.msgpack'
    final = serialization.from_bytes(loaded.params, final_path.read_bytes())
    assert any(
        not np.array_equal(loaded.params['params'][name], value)
        for name, value in final['params'].items()
    )
