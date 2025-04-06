import os
from collections import namedtuple

import numpy as np
import pandas as pd

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
morgan_generator = GetMorganGenerator(radius=1, fpSize=512)

def create_fingerprints(smiles):
    return np.stack([
        morgan_generator.GetFingerprintAsNumPy(MolFromSmiles(smile))
        for smile in smiles
    ])

def load_immuno(datadir, x_type='embeddings'):

    data = {'train': {}, 'validation': {}, 'test': {}}

    # SMILES
    data['train']['smiles'] = pd.read_csv(os.path.join(datadir, 'train/smiles.csv')).to_numpy().squeeze()
    data['validation']['smiles'] = pd.read_csv(os.path.join(datadir, 'validation/smiles.csv')).to_numpy().squeeze()
    data['test']['smiles'] = pd.read_csv(os.path.join(datadir, 'test/smiles.csv')).to_numpy().squeeze()
    
    # Unimol Embeddings
    if x_type == 'embeddings':
        data['train']['x'] = np.load(os.path.join(datadir, 'train/unimol_embeddings.npy'))
        data['validation']['x'] = np.load(os.path.join(datadir, 'validation/unimol_embeddings.npy'))
        data['test']['x'] = np.load(os.path.join(datadir, 'test/unimol_embeddings.npy'))
    # Morgan Fingerprints
    if x_type == 'fingerprints':
        data['train']['x'] = create_fingerprints(data['train']['smiles'])
        data['validation']['x'] = create_fingerprints(data['validation']['smiles'])
        data['test']['x'] = create_fingerprints(data['test']['smiles'])

    # Immunomodulation Targets
    data['train']['y'] = pd.read_csv(os.path.join(datadir, 'train/y.csv')).to_numpy()
    data['validation']['y'] = pd.read_csv(os.path.join(datadir, 'validation/y.csv')).to_numpy()
    data['test']['y'] = pd.read_csv(os.path.join(datadir, 'test/y.csv')).to_numpy()

    # Read property labels
    with open(os.path.join(datadir, 'train/y.csv'), "r") as f:
        properties = f.readline().strip().split(',')

    # RDKit Targets
    try:
        data['train']['y_rdkit'] = pd.read_csv(os.path.join(datadir, 'train/y_rdkit.csv')).to_numpy()
        data['validation']['y_rdkit'] = pd.read_csv(os.path.join(datadir, 'validation/y_rdkit.csv')).to_numpy()
        data['test']['y_rdkit'] = pd.read_csv(os.path.join(datadir, 'test/y_rdkit.csv')).to_numpy()
        #with open(os.path.join(datadir, 'train/y_rdkit.csv'), "r") as f:
        #    properties += f.readline().split()
    except: None

    # Immunomodulation Standard Errors
    try:
        data['train']['y_std'] = pd.read_csv(os.path.join(datadir, 'train/std.csv')).to_numpy()
        data['validation']['y_std'] = pd.read_csv(os.path.join(datadir, 'validation/std.csv')).to_numpy()
        data['test']['y_std'] = pd.read_csv(os.path.join(datadir, 'test/std.csv')).to_numpy()
    except: None

    data['properties'] = properties

    return data

def to_tensor(data, device='cpu'):
    
    import torch
    
    for split in data.keys():
        if type(data[split]) is not dict: continue
        for item in data[split].keys():
            try:
                data[split][item] = torch.tensor(
                    data[split][item], 
                    dtype=torch.float32
                ).detach().to(device)
            except:
                continue

    return data

def to_namedtuple(data):

    Splits = namedtuple('Splits', ['train', 'validation', 'test', 'properties'])
    Data = namedtuple('Data', data['train'].keys())

    train = Data(**data['train'])
    validation = Data(**data['validation'])
    test = Data(**data['test'])

    return Splits(train=train, validation=validation, test=test, properties=data['properties'])
