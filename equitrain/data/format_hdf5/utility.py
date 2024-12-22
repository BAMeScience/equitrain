
from typing import Dict, List, Optional, Sequence, Tuple

from equitrain.data import Configurations


def write_value(value):
    return value if value is not None else "None"


def save_dataset_as_HDF5(dataset: List, out_name: str) -> None:
    with h5py.File(out_name, "w") as f:
        for i, data in enumerate(dataset):
            grp = f.create_group(f"config_{i}")
            grp["num_nodes"] = data.num_nodes
            grp["edge_index"] = data.edge_index
            grp["positions"] = data.positions
            grp["shifts"] = data.shifts
            grp["unit_shifts"] = data.unit_shifts
            grp["cell"] = data.cell
            grp["node_attrs"] = data.node_attrs
            grp["weight"] = data.weight
            grp["energy_weight"] = data.energy_weight
            grp["forces_weight"] = data.forces_weight
            grp["stress_weight"] = data.stress_weight
            grp["virials_weight"] = data.virials_weight
            grp["forces"] = data.forces
            grp["energy"] = data.energy
            grp["stress"] = data.stress
            grp["virials"] = data.virials
            grp["dipole"] = data.dipole
            grp["charges"] = data.charges


def save_AtomicData_to_HDF5(data, i, h5_file) -> None:
    grp = h5_file.create_group(f"config_{i}")
    grp["num_nodes"] = data.num_nodes
    grp["edge_index"] = data.edge_index
    grp["positions"] = data.positions
    grp["shifts"] = data.shifts
    grp["unit_shifts"] = data.unit_shifts
    grp["cell"] = data.cell
    grp["node_attrs"] = data.node_attrs
    grp["weight"] = data.weight
    grp["energy_weight"] = data.energy_weight
    grp["forces_weight"] = data.forces_weight
    grp["stress_weight"] = data.stress_weight
    grp["virials_weight"] = data.virials_weight
    grp["forces"] = data.forces
    grp["energy"] = data.energy
    grp["stress"] = data.stress
    grp["virials"] = data.virials
    grp["dipole"] = data.dipole
    grp["charges"] = data.charges


def save_configurations_as_HDF5(configurations: Configurations, i, h5_file) -> None:
    grp = h5_file.create_group(f"config_batch_{i}")
    for i, config in enumerate(configurations):
        subgroup_name = f"config_{i}"
        subgroup = grp.create_group(subgroup_name)
        subgroup["atomic_numbers"] = write_value(config.atomic_numbers)
        subgroup["positions"] = write_value(config.positions)
        subgroup["energy"] = write_value(config.energy)
        subgroup["forces"] = write_value(config.forces)
        subgroup["stress"] = write_value(config.stress)
        subgroup["virials"] = write_value(config.virials)
        subgroup["dipole"] = write_value(config.dipole)
        subgroup["charges"] = write_value(config.charges)
        subgroup["cell"] = write_value(config.cell)
        subgroup["pbc"] = write_value(config.pbc)
        subgroup["weight"] = write_value(config.weight)
        subgroup["energy_weight"] = write_value(config.energy_weight)
        subgroup["forces_weight"] = write_value(config.forces_weight)
        subgroup["stress_weight"] = write_value(config.stress_weight)
        subgroup["virials_weight"] = write_value(config.virials_weight)
        subgroup["config_type"] = write_value(config.config_type)
