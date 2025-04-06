import torch
import torch.nn as nn
import re

def expand_stack(stack):
    '''Expand shorthand stack. For example, 4LC -> LCLCLCLC'''
    expanded, n = '', 1
    parts = re.findall(r'\d+|[^\d]+', stack)
    for part in parts:
        if part.isdigit(): n = int(part)
        else: expanded += part * n

    return expanded

class MultiheadAttention(nn.Module):
    def __init__(self, E, H):
        super().__init__()

        assert E % H == 0

        self.E, self.H = E, H
        self.scale = (E // H) ** -0.5

        self.QKV = nn.Linear(E, E * 3, bias=False)
        self.out_map = nn.Linear(E, E, bias=False)

    def forward(self, embeddings, mask=None, bias=None):

        B, L, E = embeddings.size() # Batch, no. Tokens, Embed dim.
        A = E // self.H # Attention dim.

        # Compute and separate Q, K, V matrices

        qkv = self.QKV(embeddings)
        qkv = qkv.reshape(B, L, self.H, 3 * A)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute masked attention pattern

        attn = q @ k.transpose(-2, -1) * self.scale
        if bias is not None:
            attn += bias
        if mask is not None: 
            attn.masked_fill_(mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)

        # Compute values

        values = attn @ v
        values = values.permute(0, 2, 1, 3) # (B, L, H, A)
        values = values.reshape(B, L, E) # E = H * A
        
        return self.out_map(values)

class TransformerBlock(nn.Module):
    def __init__(self, E, H, dropout):
        super().__init__()
        
        self.attention = MultiheadAttention(E, H)
        self.norm_1 = nn.LayerNorm(E)
        self.mlp = nn.Sequential(
            nn.Linear(E, E * 4), 
            nn.ReLU(), 
            nn.Linear(E * 4, E)
        )
        self.norm_2 = nn.LayerNorm(E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, padding_mask, causal_mask=None, bias=None):

        # Attention residual block

        x0 = self.norm_1(x0)
        x1 = self.attention(x0, causal_mask, bias)
        x1 = self.dropout(x1) 
        x2 = x1 + x0

        # MLP residual block

        x2 = self.norm_2(x2)
        x3 = self.mlp(x2)
        x3.masked_fill_(padding_mask, 0)
        x3 = self.dropout(x3)
        x4 = x3 + x2

        return x4

class GraphTransformer(nn.Module):
    '''
    Transformer with alternating local and global masked self-attention. 
    '''
    def __init__(self, numerical_features, categorical_features, E, H, stack, out_features, dropout):
        super().__init__()

        self.E, self.H = E, H
        self.stack = expand_stack(stack)

        # Embedding layers
        self.numerical_embed = nn.Linear(numerical_features, E, bias=False)
        self.categorical_embeds = nn.ModuleList([
            nn.Embedding(n_categories, E, padding_idx=0) 
            for n_categories in categorical_features
        ])

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(E, H, dropout)
            for _ in range(len(stack))
        ])

        # Out map
        self.norm = nn.LayerNorm(E)
        self.out_map = nn.Linear(E, out_features)

    def forward(self, nodes_numerical, nodes_categorical, d, adj, padding):

        B, L, _ = nodes_categorical.size()

        # Create causal and padding masks

        padding_mask = padding.unsqueeze(-1).expand(B, L, self.E)
        padding_causal_mask = (padding.unsqueeze(-2) | padding.unsqueeze(-1)).unsqueeze(1)
        graph_causal_mask = ~adj.unsqueeze(1)
        diag_causal_mask = torch.diag(torch.ones(L)).bool().expand_as(padding_causal_mask).to(padding.device)
        if 'M' in self.stack:
            mixed_causal_mask = torch.concat((
                (padding_causal_mask | diag_causal_mask).expand(B, self.H // 2, L, L), 
                graph_causal_mask.expand(B, self.H // 2, L, L), 
            ), dim=1)

        # Create biases

        coulomb_bias = -2 * torch.log(d).unsqueeze(1)
        electro_bias = -2 * torch.log(d + 0.2).unsqueeze(1)
        potential_bias = -torch.log(d).unsqueeze(1)
        if 'M' in self.stack:
            mixed_bias = torch.concat((
                coulomb_bias.expand(B, self.H // 2, L, L), 
                torch.zeros(B, self.H // 2, L, L).to(coulomb_bias.device), 
            ), dim=1)

        # Forward Pass

        x = sum(embed(nodes_categorical[:, :, i]) for i, embed in enumerate(self.categorical_embeds))
        x += self.numerical_embed(nodes_numerical)

        for block_type, transformer_block in zip(self.stack, self.transformer_blocks):
            if block_type == 'L': 
                x = transformer_block(x, padding_mask, graph_causal_mask)
            elif block_type == 'G': 
                x = transformer_block(x, padding_mask, padding_causal_mask)
            elif block_type == 'C': 
                x = transformer_block(x, padding_mask, padding_causal_mask | diag_causal_mask, coulomb_bias)
            elif block_type == 'P': 
                x = transformer_block(x, padding_mask, padding_causal_mask | diag_causal_mask, potential_bias)
            elif block_type == 'D':
                x = transformer_block(x, padding_mask, padding_causal_mask | diag_causal_mask, d.unsqueeze(1))
            elif block_type == 'M':
                x = transformer_block(x, padding_mask, mixed_causal_mask, mixed_bias)
            elif block_type == 'E':
                x = transformer_block(x, padding_mask, padding_causal_mask, electro_bias)

        x = self.norm(x)
        x = x.mean(dim=1) # (B, E)

        return self.out_map(x)
