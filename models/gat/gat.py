import torch
import torch.nn as nn

class GATConv(nn.Module):
    def __init__(self, E, H, leakage):
        super().__init__()

        assert E % H == 0

        self.E, self.H, self.A = E, H, E // H

        self.QK = nn.Linear(E, E * 2, bias=False)
        self.attn_map = nn.Linear(self.A, 1, bias=False)
        self.lrelu = nn.LeakyReLU(leakage)
        self.out_map = nn.Linear(E, E, bias=False)

    def forward(self, embeddings, adj):

        B, L, E = embeddings.size() # Batch, Tokens, Embed dim.

        qk = self.QK(embeddings)
        qk = qk.reshape(B, L, self.H, 2 * self.A)
        qk = qk.permute(0, 2, 1, 3)
        q, k = qk.chunk(2, dim=-1)

        attn = q.unsqueeze(2) + k.unsqueeze(3)
        attn = self.lrelu(attn)
        attn = self.attn_map(attn).squeeze(-1)
        attn.masked_fill_(~adj.unsqueeze(1), torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)
        
        values = attn @ k
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(B, L, E)

        return self.out_map(values)

class GATBlock(nn.Module):
    def __init__(self, E, H, leakage, dropout):
        super().__init__()
        
        self.gat_conv = GATConv(E, H, leakage)
        self.norm_0 = nn.LayerNorm(E)
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
        x1 = self.gat_conv(x0, adj)
        x1 = self.dropout(x1) 
        x2 = x1 + x0

        # MLP residual block

        x2 = self.norm_1(x2)
        x3 = self.mlp(x2)
        x3.masked_fill_(padding_mask, 0)
        x3 = self.dropout(x3)
        x4 = x3 + x2

        return x4

class GATTransformer(nn.Module):
    def __init__(self, numerical_features, categorical_features, E, H, stack, out_features, leakage, dropout):
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
        self.gat_blocks = nn.ModuleList([
            GATBlock(E, H, leakage, dropout)
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

        for gat_block in self.gat_blocks: x = gat_block(x, adj, padding_mask)
        
        x = self.norm(x)
        x = x.mean(dim=1) # (B, E)

        return self.out_map(x)
