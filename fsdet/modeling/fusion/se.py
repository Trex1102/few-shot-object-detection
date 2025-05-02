import torch
import torch.nn as nn
import torch.nn.functional as F

class SEFusion(nn.Module):
    """
    Squeeze-and-Excitation fusion: uses branch embeddings to reweight RoI features.
    """
    def __init__(self, channel_dim, emb_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim*2, channel_dim // 16)
        self.fc2 = nn.Linear(channel_dim // 16, channel_dim)

    def forward(self, roi_feat_map, ctx_emb, shp_emb):
        b = ctx_emb.size(0)
        emb = torch.cat([ctx_emb, shp_emb], dim=1)
        x = F.relu(self.fc1(emb))
        w = torch.sigmoid(self.fc2(x)).view(b, -1, 1, 1)
        return roi_feat_map * w