import ase
import numpy as np
import torch
import torch_geometric

from typing import List

from torch_geometric.data.collate import collate

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from equitrain.model           import get_model
from equitrain.data.loaders    import get_dataloader
from equitrain.data.graphs     import AtomsToGraphs
from equitrain.data.statistics import AtomicNumberTable
from equitrain.utility         import set_dtype


def predict_structures(model: torch.nn.Module, structure_list: List[Structure], z_table : AtomicNumberTable, max_neighbors=200, r_max = None, num_workers = 1, pin_memory = False, batch_size = 12, device = None) -> List[torch.Tensor]:
    """Predict energy, forces, and stress of a structure"""

    atoms_list = [ AseAtomsAdaptor.get_atoms(structure) for structure in structure_list ]

    return predict_atoms(model, atoms_list, z_table, max_neighbors = max_neighbors, r_max = r_max, num_workers = num_workers, pin_memory = pin_memory, batch_size = batch_size, device = device)


def predict_atoms(model: torch.nn.Module, atoms_list: List[ase.Atoms], z_table : AtomicNumberTable, max_neighbors=200, r_max = None, num_workers = 1, pin_memory = False, batch_size = 12, device = None) -> List[torch.Tensor]:
    """Predict energy, forces, and stress of a structure"""

    if hasattr(model, 'cutoff'):
        r_max = model.cutoff
    if hasattr(model, 'max_radius'):
        r_max = model.max_radius

    if r_max is None:
        raise ValueError('Could not determine r_max value')

    atoms_to_graphs = AtomsToGraphs(
        z_table,
        max_neigh=max_neighbors,
        radius=r_max,
        r_energy=False,
        r_forces=False,
        r_stress=False,
        r_distances=True,
        r_edges=True,
        r_fixed=False,
        r_pbc=True,
    )

    graph_list = [ atoms_to_graphs.convert(atom) for atom in atoms_list ]

    return predict_graphs(model, graph_list, num_workers = num_workers, pin_memory = pin_memory, batch_size = batch_size, device = device)


def predict_graphs(model: torch.nn.Module, graph_list: List[torch_geometric.data.data.Data], num_workers = 1, pin_memory = False, batch_size = 12, device = None) -> List[torch.Tensor]:

    data_loader = torch_geometric.loader.DataLoader(
        dataset=graph_list,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    r_energy = torch.empty((0), device=device)
    r_force  = torch.empty((0, 3), device=device)
    r_stress = torch.empty((0, 3, 3), device=device)

    for data in data_loader:

        data = data.to(device)

        energy, force, stress = model(data)

        r_energy = torch.cat((r_energy, energy), dim=0)
        r_force  = torch.cat((r_force , force ), dim=0)
        r_stress = torch.cat((r_stress, stress), dim=0)

    return r_energy, r_force, r_stress


def _predict(args, device=None):

    set_dtype(args.dtype)

    r_energy = torch.empty((0), device=device)
    r_force  = torch.empty((0, 3), device=device)
    r_stress = torch.empty((0, 3, 3), device=device)

    data_loader, r_max = get_dataloader(args.predict_file, args)

    model = get_model(r_max, args)
    model = model.to(device)

    for step, data in enumerate(data_loader):

        data = data.to(device)

        energy, force, stress = model(data)

        r_energy = torch.cat((r_energy, energy), dim=0)
        r_force  = torch.cat((r_force , force ), dim=0)
        r_stress = torch.cat((r_stress, stress), dim=0)

    return r_energy, r_force, r_stress


def predict(args):

    if args.predict_file is None:
        raise ValueError("--predict-file is a required argument")
    if args.statistics_file is None:
        raise ValueError("--statistics-file is a required argument")
    if args.load_checkpoint_model is None:
        raise ValueError("--load-checkpoint-model is a required argument")

    return _predict(args)
