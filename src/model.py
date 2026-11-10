"""Model definition used by the demo (inference only).

This is the same MMADL class as in notebooks/mmadl-model.ipynb.
"""
import torch
import torch.nn as nn


class ViewEncoder(nn.Module):
    def __init__(self, in_dim, d, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d),
                                 nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MMADL(nn.Module):
    """Per drug: 5 view encoders -> view embeddings -> self-attention over the views
    -> drug vector. Per pair: [z1, z2, z1*z2] -> MLP classifier.
    Encoders are shared between drug 1 and drug 2."""

    def __init__(self, drug_views, num_classes, d=256, heads=4, dropout=0.3, view_drop=0.1):
        super().__init__()
        self.names = list(drug_views)
        for n, x in drug_views.items():
            self.register_buffer(f"x_{n}", x)
        self.encoders = nn.ModuleDict(
            {n: ViewEncoder(x.shape[1], d, dropout) for n, x in drug_views.items()})
        self.view_emb = nn.Parameter(torch.randn(len(self.names), d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(2 * d, d))
        self.norm2 = nn.LayerNorm(d)
        self.drug_proj = nn.Sequential(nn.Linear(len(self.names) * d, d),
                                       nn.LayerNorm(d), nn.GELU())
        self.view_drop = view_drop
        self.decoder = nn.Sequential(
            nn.Linear(3 * d, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes))

    def encode_drugs(self):
        tokens = torch.stack(
            [self.encoders[n](getattr(self, f"x_{n}")) for n in self.names], dim=1)
        tokens = tokens + self.view_emb
        if self.training and self.view_drop > 0:
            keep = (torch.rand(tokens.shape[0], tokens.shape[1], 1,
                               device=tokens.device) > self.view_drop).float()
            tokens = tokens * keep
        a, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + a)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return self.drug_proj(tokens.flatten(1))

    def pair_features(self, i1, i2):
        z = self.encode_drugs()
        z1, z2 = z[i1], z[i2]
        return torch.cat([z1, z2, z1 * z2], dim=1)

    def forward(self, i1, i2):
        return self.decoder(self.pair_features(i1, i2))
        
    def embed(self, i1, i2): 
        return self.decoder[:-1](self.pair_features(i1, i2))