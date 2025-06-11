import torch

from rdkit import Chem
from rdkit.Chem import MolFromSmiles

from torch.nn.utils.rnn import pad_sequence

atom_categories = {
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ], 
    'hybridization': [
        'UNSPECIFIED', 
        'S', 
        'SP', 
        'SP2', 
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

def get_numerical_features(atom):
    return [
        atom.GetAtomicNum(), 
        atom.GetTotalDegree(), 
        atom.GetFormalCharge(), 
        atom.GetTotalNumHs(), 
        atom.GetNumRadicalElectrons(), 
    ]
    
def get_categorical_features(atom):
    return [
        atom_categories['chirality'].index(str(atom.GetChiralTag())), 
        atom_categories['hybridization'].index(str(atom.GetHybridization())), 
        atom_categories['is_aromatic'].index(atom.GetIsAromatic()), 
        atom_categories['is_in_ring'].index(atom.IsInRing())
    ]

def smiles_to_graphs(smiles, device='cpu'):

    batch_nodes_numerical = []
    batch_nodes_categorical = []
    batch_edges = []
    batch_padding = []
    
    for smile in smiles:
        mol = MolFromSmiles(smile)

        # Featurize Nodes
        numerical_nodes = torch.tensor([
            get_numerical_features(atom) for atom in mol.GetAtoms()
            if atom.GetSymbol() != "H"
        ])
        batch_nodes_numerical.append(numerical_nodes)
        
        categorical_nodes = torch.tensor([
            get_categorical_features(atom) for atom in mol.GetAtoms()
            if atom.GetSymbol() != "H"
        ]) + 1
        batch_nodes_categorical.append(categorical_nodes)
        
        batch_padding.append(torch.zeros(categorical_nodes.shape[0], dtype=torch.bool))

        # Create edge list
        start_idxs = [bond.GetBeginAtomIdx() for bond in mol.GetBonds()]
        end_idxs = [bond.GetEndAtomIdx() for bond in mol.GetBonds()]
        edges = torch.tensor([
            start_idxs + end_idxs + list(range(mol.GetNumAtoms())), 
            end_idxs + start_idxs + list(range(mol.GetNumAtoms()))
        ])
        batch_edges.append(edges)

    # Max no. of tokens
    
    n_tokens = max(len(mol) for mol in batch_nodes_categorical)

    batch_nodes_numerical = pad_sequence(
        batch_nodes_numerical, 
        batch_first=True, padding_value=0, padding_side='right'
    ).to(device)

    batch_nodes_categorical = pad_sequence(
        batch_nodes_categorical, 
        batch_first=True, padding_value=0, padding_side='right'
    ).to(device)

    batch_padding = pad_sequence(
        batch_padding, 
        batch_first=True, padding_value=True, padding_side='right'
    ).to(device)

    batch_edges = torch.stack([
        torch.sparse_coo_tensor(
            edges, torch.ones(edges.shape[1], dtype=torch.bool), (n_tokens, n_tokens)
        ).to_dense()
        for edges in batch_edges
    ]).to(device)

    return batch_nodes_numerical, batch_nodes_categorical, batch_edges, batch_padding

