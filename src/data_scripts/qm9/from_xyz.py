import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

import os

outdir = '/scratch/midway3/jshe/data/qm9/raw/'
xyzdir = '/scratch/midway3/jshe/data/qm9/xyz/'

bad_samples = set([
    21725, 87037, 59827, 117523, 128113, 
    129053, 129152, 129158, 130535, 6620, 
    59818, 21725, 59827, 128113, 129053, 
    129152, 130535, 6620, 59818
])

# Loop through XYZ files

all_coordinates = []
all_conformers = []
all_partial_charges = []
all_atoms = []
all_smiles = []
all_properties = []

for i, fname in enumerate(os.listdir(xyzdir)):

    # Read XYZ file

    fname = os.path.join(xyzdir, fname)
    with open(fname) as f: lines = f.readlines()

    # Read number of atoms

    n_atoms = int(lines[0])

    # Read properties and SMILES

    tag, mol_id, *properties = lines[1].split()
    if int(mol_id) in bad_samples: continue
    smile, relaxed_smile = lines[-2].split()

    # Create conformer

    mol = Chem.MolFromSmiles(relaxed_smile)
    if mol is None: continue
    mol = Chem.AddHs(mol)
    embed = AllChem.EmbedMolecule(mol, randomSeed=16)
    if embed != 0: continue
    AllChem.UFFOptimizeMolecule(mol)

    # Read out conformer coordinates

    j, conformers = 0, np.full((9, 3), np.inf)
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'H': continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        conformers[j] = pos.x, pos.y, pos.z
        j += 1

    # Read out coordinates, partial charges, and atoms from XYZ file

    coordinates = np.full((29, 3), np.inf)
    partial_charges = np.zeros(29)
    atoms = ''
    for j in range(2, n_atoms + 2):
        atom, *(coordinates[j-2]), partial_charges[j-2] = lines[j].replace('*^', 'e').split()
        atoms += atom

    # Append to lists

    all_smiles.append(smile)
    all_properties.append(properties)
    all_coordinates.append(coordinates)
    all_conformers.append(conformers)
    all_partial_charges.append(partial_charges)
    all_atoms.append(atoms)

# Save data as .npy files

np.save(outdir + 'coordinates.npy', np.array(all_coordinates).astype(float))
np.save(outdir + 'conformers.npy', np.array(all_conformers).astype(float))
np.save(outdir + 'partial_charges.npy', np.array(all_partial_charges).astype(float))
np.save(outdir + 'properties.npy', np.array(all_properties).astype(float))
np.save(outdir + 'atoms.npy', np.array(all_atoms))
np.save(outdir + 'smiles.npy', np.array(all_smiles))
