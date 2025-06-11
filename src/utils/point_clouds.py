import torch
from torch.nn.utils.rnn import pad_sequence

from rdkit.Chem import GetPeriodicTable

ptable = GetPeriodicTable()

def get_numerical_features(atom):
    return [
        ptable.GetAtomicNumber(atom),
        ptable.GetNOuterElecs(atom),
        ptable.GetRvdw(atom),
        ptable.GetRcovalent(atom)
    ]

def featurize_atoms(atoms_arr, device='cpu'):
    '''
    Featurize a list of atoms corresponding to a molecule.

    Args:
        atoms: List[str]

    Returns:
        List[List[float]]
    '''

    features_list = [
        torch.tensor([get_numerical_features(atom) for atom in atoms], device=device)
        for atoms in atoms_arr
    ]
    features_tensor = pad_sequence(features_list, batch_first=True, padding_value=0)

    padding_list = [
        torch.zeros(len(atoms), dtype=torch.bool, device=device)
        for atoms in atoms_arr
    ]
    padding_tensor = pad_sequence(padding_list, batch_first=True, padding_value=True)

    return features_tensor, padding_tensor

def collate_point_clouds(batch):
    '''
    Collate a DataLoader outputs corresponding to point clouds.

    Args:
        batch: List(Tuple(str, ...))

    Returns:
        Tuple(torch.Tensor, torch.Tensor, ...)
    '''

    atoms_batch, coordinates_batch, y_batch = zip(*batch)

    atoms, padding = featurize_atoms(atoms_batch)
    coordinates = torch.tensor(coordinates_batch)[:, :atoms.shape[1], :]
    y = torch.tensor(y_batch)

    return atoms, padding, coordinates, y
