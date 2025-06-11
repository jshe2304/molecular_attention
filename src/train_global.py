import os
import sys
from datetime import datetime

from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.datasets import *
from utils.point_clouds import *

from models.global_transformer import Transformer


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

E, H, D, block_type, *_ = sys.argv[1:]
E, H, D = int(E), int(H), int(D)

###################
# Logging directory
###################

logfile = f'./logs/qm9_scaffold/E{E}_H{H}{block_type}{D}/'
logfile += datetime.now().strftime("%m_%d_%H_%M_%S")
logfile += '_'
logfile += os.environ.get("SLURM_ARRAY_TASK_ID")
logfile += '.csv'

os.makedirs(os.path.dirname(logfile), exist_ok=True)
with open(logfile, 'w') as f:
    f.write('train_mse,validation_mse,validation_r2\n')

######
# Data
######

datadir = '/scratch/midway3/jshe/data/qm9/scaffolded/'
fnames = [
    'atoms.npy',
    'coordinates.npy',
    'norm_y.npy', 
]

train_dataset = NPYDataset(*[
    datadir + f'train_{fname}' for fname in fnames
])
validation_dataset = NPYDataset(*[
    datadir + f'validation_{fname}' for fname in fnames
])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=2048, shuffle=True)

n_properties = train_dataset.datas[-1].shape[1]

#######
# Model
#######

model = Transformer(
    in_features=4, 
    out_features=n_properties, 
    E=E, H=H, D=D, 
    block_type=block_type,
    dropout=0.1, 
).to(device)

#######
# Train
#######

optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
mse = nn.MSELoss()

for epoch in range(64):
    for smiles, r, y_true in train_dataloader:
        model.train()
        optimizer.zero_grad()

        # Create graphs

        nodes_numerical, nodes_categorical, adj, padding = smiles_to_graphs(smiles, device=device)

        # Forward pass
        
        y_pred = model(
            nodes_numerical.float(), nodes_categorical, 
            adj, padding, r.to(device)
        )
        loss = mse(y_pred, y_true.to(device))
        loss.backward()
        optimizer.step()

        print(loss)

    # Log train statistics

    model.eval()
    with torch.no_grad():

        smiles, r, y_true = next(iter(validation_dataloader))
        nodes_numerical, nodes_categorical, adj, padding = smiles_to_graphs(smiles, device=device)
        y_pred = model(
            nodes_numerical.float(), nodes_categorical, 
            adj, padding, r.to(device), 
        )
        
        validation_loss = float(mse(y_pred, y_true.to(device)))
        validation_score = float(r2_score(y_true.cpu(), y_pred.cpu()))

    # Write to log

    with open(logfile, 'a') as f:
        f.write(f'{float(loss)},{validation_loss},{validation_score}\n')

# Save model

weights_file = f'./weights/qm9_scaffold/E{E}_H{H}_{stack}.pt'
os.makedirs(os.path.dirname(weights_file), exist_ok=True)
torch.save(model.state_dict(), weights_file)
