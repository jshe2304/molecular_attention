import torch
import torch.nn as nn

class GINOperator(nn.Module):
    def __init__(self, E, H, **kwargs):
        super().__init__()
        assert E % H == 0
        
        self.mlp = nn.Sequential(
            nn.Linear(E, E * H // 2), 
            nn.ReLU(), 
            nn.Linear(E * H // 2, E)
        )

    def forward(self, embeddings, adj, **kwargs):
        return self.mlp(adj.float() @ embeddings)

class GATv2Operator(nn.Module):
    def __init__(self, E, H, **kwargs):
        super().__init__()
        assert E % H == 0

        self.E, self.H, self.A = E, H, E // H

        self.QK = nn.Linear(E, E * 2, bias=False)
        self.attn_map = nn.Linear(self.A, 1, bias=False)
        self.lrelu = nn.LeakyReLU(kwargs.get('leakage', 0.1))
        self.out_map = nn.Linear(E, E, bias=False)

    def forward(self, embeddings, attn_mask, **kwargs):

        B, L, E = embeddings.size() # Batch, Tokens, Embed dim.

        qk = self.QK(embeddings)
        qk = qk.reshape(B, L, self.H, 2 * self.A)
        qk = qk.permute(0, 2, 1, 3)
        q, k = qk.chunk(2, dim=-1)

        attn = q.unsqueeze(2) + k.unsqueeze(3)
        attn = self.lrelu(attn)
        attn = self.attn_map(attn).squeeze(-1)
        attn.masked_fill_(attn_mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)
        
        values = attn @ k
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(B, L, E)

        return self.out_map(values)

class GATv3Operator(nn.Module):
    def __init__(self, E, H, **kwargs):
        super().__init__()
        assert E % H == 0

        self.E, self.H, self.A = E, H, E // H

        self.QKV = nn.Linear(E, E * 3, bias=False)
        self.attn_map = nn.Linear(self.A, 1, bias=False)
        self.lrelu = nn.LeakyReLU(kwargs.get('leakage', 0.1))
        self.out_map = nn.Linear(E, E, bias=False)

    def forward(self, embeddings, attn_mask, **kwargs):

        B, L, E = embeddings.size() # Batch, Tokens, Embed dim.

        qkv = self.QKV(embeddings)
        qkv = qkv.reshape(B, L, self.H, 3 * self.A)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        attn = q.unsqueeze(2) + k.unsqueeze(3)
        attn = self.lrelu(attn)
        attn = self.attn_map(attn).squeeze(-1)
        attn.masked_fill_(attn_mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)
        
        values = attn @ v
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(B, L, E)

        return self.out_map(values)

class SDPAOperator(nn.Module):
    def __init__(self, E, H, **kwargs):
        super().__init__()

        assert E % H == 0

        self.E, self.H = E, H
        self.scale = (E // H) ** -0.5

        self.QKV = nn.Linear(E, E * 3, bias=False)
        self.out_map = nn.Linear(E, E, bias=False)

    def forward(self, embeddings, attn_mask=None, attn_bias=0):

        B, L, E = embeddings.size() # Batch, no. Tokens, Embed dim.
        A = E // self.H # Attention dim.

        # Compute Q, K, V matrices

        qkv = self.QKV(embeddings)
        qkv = qkv.reshape(B, L, self.H, 3 * A)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute masked attention pattern

        attn = q @ k.transpose(-2, -1) * self.scale
        attn += attn_bias
        if attn_mask is not None: 
            attn.masked_fill_(attn_mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)

        # Compute values

        values = attn @ v
        values = values.permute(0, 2, 1, 3) # (B, L, H, A)
        values = values.reshape(B, L, E) # E = H * A
        
        return self.out_map(values)