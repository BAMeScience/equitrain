# %% Imports
import json
import numpy as np
import random

from ase.io import write
from ase.units import GPa
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

# %%

__author__ = "Philipp Benner"
__date__ = "2023-08-11"

# %%

class WeightedRandomizer:
    def __init__ (self, weights, seed=42):
        random.seed(seed)
        self.__max = .0
        self.__weights = []
        for value, weight in weights.items ():
            self.__max += weight
            self.__weights.append ( (self.__max, value) )

    def __call__(self):
        r = random.random () * self.__max
        for ceil, value in self.__weights:
            if ceil > r: return value

# %%

with open('MPtrj_2022.9_full.json') as f:
    js = json.load(f)

# Random assignment to train, valid, or test set
rand = WeightedRandomizer({'train': 0.95, 'valid': 0.05, 'test': 0.00})

# A dictionary used to put all structures with the same composition
# to go into the same train, valid, or test set
selection_dict = {}

r_train = []
r_valid = []
r_test  = []

for _, values in tqdm(js.items(), desc='Converting data', total=len(js)):

    # This variable determines if the trajectory goes into train, valid, or
    # test set
    selection = None

    for submid, subvalues in values.items():
        atoms = AseAtomsAdaptor.get_atoms(
            Structure.from_dict(subvalues['structure']),
            info={'config_type'     : 'Default',
                  'energy'          : subvalues['uncorrected_total_energy'],
                  'energy_corrected': subvalues['corrected_total_energy'],
                  'stress'          : -np.array(subvalues['stress']) * 1e-1 * GPa })
        atoms.arrays['forces'] = np.array(subvalues['force'])

        # Structures with the same composition must go into the same train, valid,
        # or test set.
        if selection is None:

            comp = atoms.get_chemical_formula(mode='reduce')

            # If this composition is unseen, select a new train, valid, or
            # test set for this composition at random
            if not comp in selection_dict:
                selection_dict[comp] = rand()

            selection = selection_dict[comp]

        if selection == 'train':
            r_train.append(atoms)
        if selection == 'valid':
            r_valid.append(atoms)
        if selection == 'test':
            r_test .append(atoms)

# %% Export result

if len(r_train) > 0:
    write('mptraj_train.xyz', r_train)
if len(r_valid) > 0:
    write('mptraj_valid.xyz', r_valid)
if len(r_test) > 0:
    write('mptraj_test.xyz' , r_test)
