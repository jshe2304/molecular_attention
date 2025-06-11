import torch
import torch.nn as nn

from .graph_operators import SDPAOperator
from .utils import *
from .misc_modules import *

class TransformerBlock(nn.Module):
    def __init__(self, E, H, dropout=0.1, bias_scaling=False):
        super().__init__()

        self.bias_scale = NegativeLinear(1, H) if bias_scaling else None
        
        self.operator = SDPAOperator(E, H)
        self.norm_1 = nn.LayerNorm(E)
        self.mlp = nn.Sequential(
            nn.Linear(E, E * 4), 
            nn.ReLU(), 
            nn.Linear(E * 4, E)
        )
        self.norm_2 = nn.LayerNorm(E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, padding_mask, attn_bias=0):

        B, L, E = x0.shape

        # Scale bias
        
        if hasattr(self, 'bias_scale'): 
            attn_bias = self.bias_scale(
                attn_bias.reshape(B, L, L, 1)
            ).permute(0, 3, 1, 2)

        # Attention block

        x0 = self.norm_1(x0)
        x1 = self.operator(x0, attn_bias=attn_bias)
        x1 = self.dropout(x1)
        x2 = x1 + x0

        # MLP block

        x2 = self.norm_2(x2)
        x3 = self.mlp(x2)
        x3.masked_fill_(padding_mask, 0)
        x3 = self.dropout(x3)
        x4 = x3 + x2

        return x4

class Transformer(nn.Module):

    def __init__(self, in_features, out_features, block_type, E, H, D, dropout=0.1):
        super().__init__()

        self.E, self.H, self.D = E, H, D
        self.block_type = block_type

        # Embedding layer
        self.embed = nn.Linear(in_features, E)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                E, H, 
                dropout=dropout, bias_scaling=(block_type == 'p')
            )
            for _ in range(D)
        ])
        
        # Out map
        self.norm = nn.LayerNorm(E)
        self.out_map = nn.Linear(E, out_features)

    def forward(self, x, padding, r):

        assert x.shape[0] == padding.shape[0] == r.shape[0]
        B, L, _ = x.shape

        # Construct log distance bias
        d = torch.norm(
            r.unsqueeze(1) - r.unsqueeze(2), 
            dim=-1
        ).unsqueeze(1)
        log_d = torch.log(d).nan_to_num(torch.finfo(r.dtype).max)

        # Create causal and padding masks

        padding_mask = padding.unsqueeze(-1).expand(B, L, self.E)
        padding_causal_mask = (padding.unsqueeze(-2) | padding.unsqueeze(-1)).unsqueeze(1)
        diag_causal_mask = torch.diag(torch.ones(L)).bool().expand_as(padding_causal_mask).to(padding.device)

        # Forward Pass

        x = self.embed(x)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(
                x, padding_mask, 
                attn_mask=padding_causal_mask | diag_causal_mask, 
                attn_bias=log_d
            )

        x = self.norm(x)
        x = x.mean(dim=1)

        return self.out_map(x)
