import os
import sys
sys.path.append('/home/jshe/prop-pred/src/data')
from data_utils.datasets import SmilesDataset
from data_utils.graphs import smiles_to_graphs

from gin import GINTransformer

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from sklearn.metrics import r2_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

log_file = './logs/gin.csv'
weights_file = './weights/gin.pt'

with open(log_file, 'w') as f:
    f.write('validation_mse,validation_r2\n')

######
# Data
######

dataset = SmilesDataset(
    smiles='/home/jshe/prop-pred/src/data/qm9/smiles.csv', 
    y='/home/jshe/prop-pred/src/data/qm9/norm_y.csv', 
    d='/home/jshe/prop-pred/src/data/qm9/distances.npy'
)
train_dataset, validation_dataset, _ = random_split(
    dataset, lengths=(0.8, 0.1, 0.1), 
    generator=torch.Generator().manual_seed(16)
)
del _

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=2048, shuffle=True)

#######
# Model
#######

hyperparameters = dict(
    numerical_features=5, categorical_features=(9+1, 8+1, 2+1, 2+1), 
    E=64, H=2, stack=8, 
    dropout=0.1, 
    out_features=dataset.n_properties, 
)

model = GINTransformer(**hyperparameters).to(device)

#######
# Train
#######

optimizer = torch.optim.Adam(model.parameters())
mse = nn.MSELoss()

for epoch in range(128):
    for smiles, d, y_true in train_dataloader:
        model.train()
        optimizer.zero_grad()

        numerical_node_features, categorical_node_features, adj, padding = smiles_to_graphs(smiles, device=device)

        # Forward pass and loss
        y_pred = model(
            numerical_node_features.float(), categorical_node_features, 
            adj.float(), padding
        )
        loss = mse(y_pred, y_true.to(device))
        loss.backward()
        optimizer.step()

    # Validation statistics
    
    model.eval()
    with torch.no_grad():
        smiles, d, y_true = next(iter(validation_dataloader))
        numerical_node_features, categorical_node_features, adj, padding = smiles_to_graphs(smiles, device=device)
        y_pred = model(
            numerical_node_features.float(), categorical_node_features, 
            adj.float(), padding
        )
        validation_loss = float(mse(y_pred, y_true.to(device)))
        validation_score = float(r2_score(y_true.cpu(), y_pred.cpu()))

    # Write to log

    with open(log_file, 'a') as f:
        f.write(f'{validation_loss},{validation_score}\n')

# Save model

torch.save(model.state_dict(), weights_file)
