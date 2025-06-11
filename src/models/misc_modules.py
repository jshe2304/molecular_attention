import torch
import torch.nn as nn

class NegativeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NegativeLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, x):
        return -nn.functional.linear(x, self.log_weight.exp())

class HybridEmbedding(nn.Module):
    def __init__(self, numerical_features, categorical_features, E):
        super().__init__()

        self.numerical_embed = nn.Linear(numerical_features, E, bias=False)
        self.categorical_embeds = nn.ModuleList([
            nn.Embedding(n_categories, E, padding_idx=0) 
            for n_categories in categorical_features
        ])

    def forward(self, nodes_numerical, nodes_categorical):
        e1 = sum(
            embed(nodes_categorical[:, :, i]) 
            for i, embed in enumerate(self.categorical_embeds)
        )
        e2 = self.numerical_embed(nodes_numerical)

        return e1 + e2