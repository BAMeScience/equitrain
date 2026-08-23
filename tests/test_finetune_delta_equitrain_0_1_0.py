from __future__ import annotations

import warnings
from pathlib import Path

import torch

from equitrain import get_args_parser_predict, predict
from equitrain.backends.torch_wrappers import MaceWrapper as TorchMaceWrapper
from equitrain.finetune.delta_torch import DeltaFineTuneWrapper
from equitrain.utility_test import mace_support

_DATA_DIR = Path(__file__).with_name('data') / 'equitrain_0_1_0_delta'
_EQUITRAIN_0_1_0_DELTA_VALUES = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
)
_NO_PAIR_MACE_TOP_LEVEL_MODULE_NAMES = (
    'node_embedding',
    'radial_embedding',
    'spherical_harmonics',
    'atomic_energies_fn',
    'interactions',
    'products',
    'readouts',
    'scale_shift',
)
_EQUITRAIN_0_1_0_TRAINABLE_BASE_PARAMETER_NAMES = (
    'model.node_embedding.linear.weight',
    'model.interactions.0.linear_up.weight',
    'model.interactions.0.conv_tp_weights.layer0.weight',
    'model.interactions.0.conv_tp_weights.layer1.weight',
    'model.interactions.0.conv_tp_weights.layer2.weight',
    'model.interactions.0.conv_tp_weights.layer3.weight',
    'model.interactions.0.linear.weight',
    'model.interactions.0.skip_tp.weight',
)

# Generated with the Equitrain 0.1.0 package, using
# tests/test_finetune_mace.py::FinetuneMaceWrapper, the legacy validation HDF5
# fixture equivalent to tests/data/equitrain_0_1_0_delta/valid.h5, and the
# deterministic no-pair MACE model built below.
_EQUITRAIN_0_1_0_VALID_ENERGY = (
    0.011251997202634811,
    0.0001427705428795889,
    0.000011426920536905527,
    0.000004193495897197863,
)


def _build_no_pair_mace_model():
    modules = mace_support._require_mace()
    AtomicData = modules.AtomicData
    config_from_atoms = modules.config_from_atoms
    torch_geometric = modules.torch_geometric
    configure_model_torch = modules.configure_model_torch

    structures = mace_support.build_structures()
    zs = [int(z) for atoms in structures for z in atoms.get_atomic_numbers()]
    statistics = mace_support.build_statistics(zs)
    args = mace_support.create_model_args(statistics)
    args.pair_repulsion = False

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
    _ = torch_geometric.batch.Batch.from_data_list(atomic_data_list).to(torch.float32)

    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(12345)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model, _ = configure_model_torch(
                args,
                train_loader=[],
                atomic_energies=statistics['atomic_energies'],
                heads=args.heads,
                z_table=statistics['atomic_numbers'],
            )
    finally:
        torch.random.set_rng_state(rng_state)

    return model.float().eval()


def _set_equitrain_0_1_0_deltas(wrapper: DeltaFineTuneWrapper) -> None:
    with torch.no_grad():
        for index, (_name, delta) in enumerate(wrapper.named_delta_parameters()):
            if index < len(_EQUITRAIN_0_1_0_DELTA_VALUES):
                delta.fill_(_EQUITRAIN_0_1_0_DELTA_VALUES[index])
            else:
                delta.zero_()


def test_delta_wrapper_matches_equitrain_0_1_0_on_legacy_fixture():
    args = get_args_parser_predict().parse_args([])
    args.predict_file = str(_DATA_DIR / 'valid.h5')
    args.batch_size = 2
    args.num_workers = 0
    args.pin_memory = False
    args.dtype = 'float32'
    args.tqdm = False
    args.forces_weight = 0.0
    args.stress_weight = 0.0
    args.batch_max_nodes = None
    args.batch_max_edges = None
    args.batch_drop = False

    base_wrapper = TorchMaceWrapper(args, _build_no_pair_mace_model())
    wrapper = DeltaFineTuneWrapper(base_wrapper, freeze_layers='2-')

    assert tuple(base_wrapper.model._modules) == _NO_PAIR_MACE_TOP_LEVEL_MODULE_NAMES
    assert [
        name for name, delta in wrapper.named_delta_parameters() if delta.requires_grad
    ] == list(_EQUITRAIN_0_1_0_TRAINABLE_BASE_PARAMETER_NAMES)

    _set_equitrain_0_1_0_deltas(wrapper)
    args.model = wrapper

    energy, _forces, _stress = predict(args)
    torch.testing.assert_close(
        energy.detach().cpu(),
        torch.tensor(_EQUITRAIN_0_1_0_VALID_ENERGY, dtype=torch.float32),
        rtol=1e-5,
        atol=1e-7,
    )
