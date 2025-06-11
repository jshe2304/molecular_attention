import torch
import torch.nn as nn

from .graph_operators import GINOperator, GATv2Operator, GATv3Operator, SDPAOperator
from .utils import *
from .misc_modules import *

operator_map = {
    'L': SDPAOperator, # Local Attention
    'G': SDPAOperator, # Global Attention
    'C': SDPAOperator, # Coulomb
    'P': SDPAOperator, # Potential
    'F': SDPAOperator, # r^-5
    'p': SDPAOperator, # Learnable r^p
    'H': SDPAOperator, # Hybrid
    'I': GINOperator, # Isomorphism
    'M': GATv2Operator, # MLP Attention
    'm': GATv3Operator # MLP Attention with Values
}

class TransformerBlock(nn.Module):
    def __init__(self, Operator, E, H, dropout=0.1, bias_scaling=False, **operator_params):
        super().__init__()

        if bias_scaling: self.bias_scale = NegativeLinear(1, H)
        
        self.operator = Operator(E, H, **operator_params)
        self.norm_1 = nn.LayerNorm(E)
        self.mlp = nn.Sequential(
            nn.Linear(E, E * 4), 
            nn.ReLU(), 
            nn.Linear(E * 4, E)
        )
        self.norm_2 = nn.LayerNorm(E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, padding_mask, **kwargs):

        B, L, E = x0.shape

        # Scale bias
        
        if hasattr(self, 'bias_scale') and 'attn_bias' in kwargs: 
            kwargs['attn_bias'] = self.bias_scale(
                kwargs['attn_bias'].reshape(B, L, L, 1)
            ).permute(0, 3, 1, 2)

        # Attention block

        x0 = self.norm_1(x0)
        x1 = self.operator(x0, **kwargs)
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

    def __init__(self, in_features, out_features, stack, E, H, dropout=0.1, **operator_params):
        super().__init__()

        self.E, self.H = E, H
        self.stack = expand_stack(stack)

        # Embedding layer
        
        self.embed = HybridEmbedding(in_features[0], in_features[1], E)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                operator_map[block_type], 
                E, H, dropout=dropout, bias_scaling=(block_type == 'p'), 
                **operator_params, 
            )
            for block_type in self.stack
        ])
        
        # Out map
        self.norm = nn.LayerNorm(E)
        self.out_map = nn.Linear(E, out_features)

    def forward(self, nodes_numerical, nodes_categorical, adj, padding, r=None):

        assert nodes_categorical.shape[0] == nodes_numerical.shape[0] == adj.shape[0] == padding.shape[0]
        assert nodes_categorical.shape[1] == nodes_numerical.shape[1] == adj.shape[1] == padding.shape[1]
        B, L, _ = nodes_categorical.shape

        # Build structural bias if needed
        
        if any(c in self.stack for c in 'CPHFp'): 
            assert r is not None
            assert r.shape[0] == B
            assert torch.all(r[:, L:] == torch.inf)

            # Crop unnecessary padding
            r = r[:, :L]

            # Construct distance matrix
            d = torch.norm(
                r.unsqueeze(1) - r.unsqueeze(2), 
                dim=-1
            ).unsqueeze(1)

            log_d = torch.log(d).nan_to_num(torch.finfo(r.dtype).max)

        # Create causal and padding masks

        padding_mask = padding.unsqueeze(-1).expand(B, L, self.E)
        padding_causal_mask = (padding.unsqueeze(-2) | padding.unsqueeze(-1)).unsqueeze(1)
        graph_causal_mask = ~adj.unsqueeze(1)
        diag_causal_mask = torch.diag(torch.ones(L)).bool().expand_as(padding_causal_mask).to(padding.device)
        if 'H' in self.stack:
            hybrid_causal_mask = torch.concat((
                (padding_causal_mask | diag_causal_mask).expand(B, self.H // 2, L, L), 
                graph_causal_mask.expand(B, self.H // 2, L, L), 
            ), dim=1)

        # Create hybrid bias if needed

        if 'H' in self.stack:
            hybrid_bias = torch.concat((
                -2 * log_d.expand(B, self.H // 2, L, L), 
                torch.zeros(B, self.H // 2, L, L).to(log_d.device), 
            ), dim=1)

        # Forward Pass

        x = self.embed(nodes_numerical, nodes_categorical)
        
        for block_type, transformer_block in zip(self.stack, self.transformer_blocks):
            if block_type == 'G': # Global
                x = transformer_block(
                    x, padding_mask, 
                    attn_mask=padding_causal_mask
                )
            elif block_type == 'I': # Isomorphism
                x = transformer_block(
                    x, padding_mask, 
                    adj=adj
                )
            elif block_type == 'C': # Coulomb
                x = transformer_block(
                    x, padding_mask, 
                    attn_mask=padding_causal_mask | diag_causal_mask, 
                    attn_bias=-2 * log_d
                )
            elif block_type == 'P': # Potential
                x = transformer_block(
                    x, padding_mask, 
                    attn_mask=padding_causal_mask | diag_causal_mask, 
                    attn_bias=-log_d
                )
            elif block_type == 'p': # Learned exponent
                x = transformer_block(
                    x, padding_mask, 
                    attn_mask=padding_causal_mask | diag_causal_mask, 
                    attn_bias=log_d
                )
            elif block_type == 'F':
                x = transformer_block(
                    x, padding_mask, 
                    attn_mask=padding_causal_mask | diag_causal_mask, 
                    attn_bias=-5 * log_d
                )
            elif block_type == 'H': # Hybrid
                x = transformer_block(
                    x, padding_mask, 
                    attn_mask=hybrid_causal_mask, 
                    attn_bias=hybrid_bias)
            else:
                x = transformer_block(
                    x, padding_mask, 
                    attn_mask=graph_causal_mask
                )

        x = self.norm(x)
        x = x.mean(dim=1)

        return self.out_map(x)
