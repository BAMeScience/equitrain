from __future__ import annotations

import ase
import torch
import torch_geometric
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from equitrain.backends.torch_model import get_model
from equitrain.backends.torch_utils import set_dtype
from equitrain.data.atomic import AtomicNumberTable
from equitrain.data.backend_torch.atoms_to_graphs import AtomsToGraphs
from equitrain.data.backend_torch.loaders import get_dataloader
from equitrain.data.configuration import niggli_reduce_inplace


def predict_graphs(
    model: torch.nn.Module,
    graph_list: list[torch_geometric.data.data.Data],
    num_workers=0,
    pin_memory=False,
    batch_size=12,
    device=None,
) -> list[torch.Tensor]:
    data_loader = torch_geometric.loader.DataLoader(
        dataset=graph_list,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    r_energy = torch.empty((0), device=device)
    r_force = torch.empty((0, 3), device=device)
    r_stress = torch.empty((0, 3, 3), device=device)

    for data in data_loader:
        y_pred = model(data)
        r_energy = torch.cat((r_energy, y_pred['energy']), dim=0)
        if y_pred['forces'] is not None:
            r_force = torch.cat((r_force, y_pred['forces']), dim=0)
        if y_pred['stress'] is not None:
            r_stress = torch.cat((r_stress, y_pred['stress']), dim=0)

    if r_force.shape[0] == 0:
        r_force = None
    if r_stress.shape[0] == 0:
        r_stress = None

    return r_energy, r_force, r_stress


def predict_atoms(
    model: torch.nn.Module,
    atoms_list: list[ase.Atoms],
    z_table: AtomicNumberTable,
    r_max: float,
    num_workers=0,
    pin_memory=False,
    batch_size=12,
    device=None,
    niggli_reduce: bool = False,
) -> list[torch.Tensor]:
    atoms_to_graphs = AtomsToGraphs(
        z_table,
        radius=r_max,
        r_energy=False,
        r_forces=False,
        r_stress=False,
        r_distances=True,
        r_edges=True,
        r_fixed=False,
        r_pbc=True,
    )

    graph_list = []
    for atom in atoms_list:
        atoms_copy = atom.copy()
        if niggli_reduce:
            niggli_reduce_inplace(atoms_copy)
        graph_list.append(atoms_to_graphs.convert(atoms_copy))

    return predict_graphs(
        model,
        graph_list,
        num_workers=num_workers,
        pin_memory=pin_memory,
        batch_size=batch_size,
        device=device,
    )


def predict_structures(
    model: torch.nn.Module,
    structure_list: list[Structure],
    z_table: AtomicNumberTable,
    r_max: float,
    num_workers=0,
    pin_memory=False,
    batch_size=12,
    device=None,
    niggli_reduce: bool = False,
) -> list[torch.Tensor]:
    atoms_list = [AseAtomsAdaptor.get_atoms(structure) for structure in structure_list]
    return predict_atoms(
        model,
        atoms_list,
        z_table,
        r_max,
        num_workers=num_workers,
        pin_memory=pin_memory,
        batch_size=batch_size,
        device=device,
        niggli_reduce=niggli_reduce,
    )


def _predict(args, device=None):
    set_dtype(args.dtype)

    model = get_model(args)
    model = model.to('cpu')
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    r_energy = torch.empty((0), device=device)
    r_force = torch.empty((0, 3), device=device)
    r_stress = torch.empty((0, 3, 3), device=device)

    data_loader = get_dataloader(
        args, args.predict_file, model.atomic_numbers, model.r_max
    )

    for data_list in data_loader:
        for data in data_list:
            if hasattr(data, 'to'):
                data = data.to(device)

            y_pred = model(data)

            r_energy = torch.cat((r_energy, y_pred['energy']), dim=0)

            if y_pred['forces'] is not None:
                r_force = torch.cat((r_force, y_pred['forces']), dim=0)

            if y_pred['stress'] is not None:
                r_stress = torch.cat((r_stress, y_pred['stress']), dim=0)

    if r_force.shape[0] == 0:
        r_force = None
    if r_stress.shape[0] == 0:
        r_stress = None

    return r_energy, r_force, r_stress


def predict(args):
    from equitrain.argparser import check_args_complete

    check_args_complete(args, 'predict')

    if args.predict_file is None:
        raise ValueError('--predict-file is a required argument')
    if args.model is None:
        raise ValueError('--model is a required argument')

    args.shuffle = False

    return _predict(args)


__all__ = [
    'predict',
    '_predict',
    'predict_graphs',
    'predict_atoms',
    'predict_structures',
]
