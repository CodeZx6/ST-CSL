import torch
import torch.nn as nn

class ContrastivePretrainModule(nn.Module):
    """Contrastive learning module for self-supervised pretraining"""
    def __init__(self, in_channels=2, embed_dim=128, temporal_len=1):
        super(ContrastivePretrainModule, self).__init__()
        self.temporal_len = temporal_len
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(temporal_len * in_channels, embed_dim, kernel_size=1),
        )
        
        self.projection_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        
        self.norm = nn.BatchNorm2d(embed_dim)
        self.spatial_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        B, T, C, H, W = x.shape[0], self.temporal_len, x.shape[1] // self.temporal_len, x.shape[2], x.shape[3]
        
        feat = self.feature_extractor(x.view(B, T * C, H, W))
        proj = self.projection_head(feat)
        proj = self.norm(proj)
        
        B, D, H, W = proj.shape
        spatial_embed = proj.permute(0, 2, 3, 1).contiguous().view(B, -1, D)
        spatial_embed = self.spatial_proj(spatial_embed)
        
        return spatial_embed

class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, embeddings):
        B, N, D = embeddings.shape
        embeddings = nn.functional.normalize(embeddings, dim=-1)
        
        loss = 0
        for b in range(B):
            anchor = embeddings[b, 0:1, :]
            sim_matrix = torch.matmul(anchor, embeddings[b].transpose(0, 1))
            sim_matrix = sim_matrix / self.temperature
            
            exp_sim = torch.exp(sim_matrix)
            pos_sim = exp_sim[:, :1]
            total_sim = exp_sim.sum(dim=1, keepdim=True)
            
            loss += -torch.log(pos_sim / total_sim).mean()
        
        return loss / B
