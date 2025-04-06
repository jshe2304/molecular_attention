import torch
import torch.nn as nn

class GINConv(nn.Module):
    def __init__(self, E, H):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(E, E * H), 
            nn.ReLU(), 
            nn.Linear(E * H, E)
        )

    def forward(self, x, adj):
        return self.mlp(adj @ x)

class GINBlock(nn.Module):
    def __init__(self, E, H, dropout):
        super().__init__()

        self.norm_0 = nn.LayerNorm(E)
        self.gin_conv = GINConv(E, H)
        self.mlp = nn.Sequential(
            nn.Linear(E, E * 4), 
            nn.ReLU(), 
            nn.Linear(E * 4, E)
        )
        self.norm_1 = nn.LayerNorm(E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, adj, padding_mask):

        # Attention residual block

        x0 = self.norm_0(x0)
        x1 = self.gin_conv(x0, adj)
        x1 = self.dropout(x1)
        x2 = x1 + x0

        # MLP residual block

        x2 = self.norm_1(x2)
        x3 = self.mlp(x2)
        x3 = self.dropout(x3)
        x3.masked_fill_(padding_mask, 0)
        x4 = x3 + x2

        return x4

class GINTransformer(nn.Module):
    def __init__(self, numerical_features, categorical_features, E, H, stack, out_features, dropout):
        super().__init__()

        self.E, self.H = E, H
        self.stack = stack

        # Embedding layers
        self.numerical_embed = nn.Linear(numerical_features, E, bias=False)
        self.categorical_embeds = nn.ModuleList([
            nn.Embedding(n_categories, E, padding_idx=0) 
            for n_categories in categorical_features
        ])

        # Transformer blocks
        self.gin_blocks = nn.ModuleList([
            GINBlock(E, H, dropout)
            for _ in range(stack)
        ])

        # Out map
        self.norm = nn.LayerNorm(E)
        self.out_map = nn.Linear(E, out_features)

    def forward(self, nodes_numerical, nodes_categorical, adj, padding):

        B, L, _ = nodes_categorical.size()

        # Make mask

        padding_mask = padding.unsqueeze(-1).expand(B, L, self.E)

        # Forward Pass

        x = sum(embed(nodes_categorical[:, :, i]) for i, embed in enumerate(self.categorical_embeds))
        x += self.numerical_embed(nodes_numerical)

        for gin_block in self.gin_blocks: x = gin_block(x, adj, padding_mask)
        
        x = self.norm(x)
        x = x.mean(dim=1) # (B, E)

        return self.out_map(x)