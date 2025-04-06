import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

import os

out_file = 'raw_qm9.csv'
xyz_folder = '/scratch/midway3/jshe/xyz'
labels = ['SMILES', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
bad_samples = set([
    21725, 87037, 59827, 117523, 128113, 
    129053, 129152, 129158, 130535, 6620, 
    59818, 21725, 59827, 128113, 129053, 
    129152, 130535, 6620, 59818
])

with open(out_file, 'w') as f:
    f.write(','.join(labels) + '\n')

# Loop through XYZ files
all_coordinates = []
for i, fname in enumerate(os.listdir(xyz_folder)):

    # Read XYZ file
    
    fname = os.path.join(xyz_folder, fname)
    with open(fname) as f: lines = f.readlines()

    # Read SMILES and targets
    
    mol_id, *targets = lines[1].split()[1:]
    if mol_id in bad_samples: continue
    smiles = lines[-2].split()[-1:]

    # Create conformer

    mol = Chem.MolFromSmiles(smiles[0])
    if mol is None: continue
    mol = Chem.AddHs(mol)
    embed = AllChem.EmbedMolecule(mol, randomSeed=16)
    if embed != 0: continue
    AllChem.UFFOptimizeMolecule(mol)

    # Read out coordinates

    j, coordinates = 0, np.full((9, 3), np.inf)
    conformer = mol.GetConformer()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H': continue
        pos = conformer.GetAtomPosition(atom.GetIdx())  # Get coordinates for each atom
        coordinates[j] = pos.x, pos.y, pos.z
        j += 1

    all_coordinates.append(coordinates)
    
    with open(out_file, 'a') as f:
        f.write(','.join(smiles + targets) + '\n')

all_coordinates = np.stack(all_coordinates)
np.save('raw_coordinates.npy', all_coordinates)