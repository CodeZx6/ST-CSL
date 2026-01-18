import numpy as np
import torch
import torch.nn.functional as F

def compute_mse(pred, target):
    """Mean Squared Error"""
    return np.mean(np.power(target - pred, 2))

def compute_rmse(pred, target):
    """Root Mean Squared Error"""
    return np.sqrt(compute_mse(pred, target))

def compute_mae(pred, target):
    """Mean Absolute Error"""
    return np.mean(np.abs(target - pred))

def compute_mape(pred, target, epsilon=1.0):
    """Mean Absolute Percentage Error"""
    target_safe = target.copy()
    target_safe[target_safe == 0] = epsilon
    return np.mean(np.abs((target - pred) / target_safe))

class ContrastiveLoss:
    """Adaptive contrastive loss with margin-based similarity"""
    def __init__(self, margin=1e-4, loss_type='softmax'):
        self.margin = margin
        self.loss_type = loss_type
    
    def __call__(self, embeddings):
        B, N, D = embeddings.shape
        device = embeddings.device
        
        total_loss = torch.tensor(0.0, device=device)
        
        for b in range(B):
            anchor_idx = torch.randint(0, N, (1,)).item()
            anchor = embeddings[b, anchor_idx:anchor_idx+1, :]
            
            # Compute similarity matrix
            sim = torch.matmul(anchor, embeddings[b].transpose(0, 1))
            
            # Compute pairwise distance for labeling
            distances = torch.sum((anchor - embeddings[b]) ** 2, dim=-1)
            labels = (distances < self.margin).long()
            
            if self.loss_type == 'softmax':
                loss = self._softmax_loss(sim, labels)
            else:
                loss = self._sigmoid_loss(sim, labels)
            
            total_loss += loss
        
        return total_loss / B
    
    def _softmax_loss(self, logits, labels):
        """Multi-label softmax contrastive loss"""
        logits = logits.float()
        labels = labels.float()
        
        logits_scaled = (1 - 2 * labels) * logits
        logits_neg = logits_scaled - labels * 1e12
        logits_pos = logits_scaled - (1 - labels) * 1e12
        
        zeros = torch.zeros_like(logits[..., :1])
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)
        
        neg_loss = torch.logsumexp(logits_neg, dim=-1)
        pos_loss = torch.logsumexp(logits_pos, dim=-1)
        
        return (neg_loss + pos_loss).mean()
    
    def _sigmoid_loss(self, logits, labels):
        """Multi-label sigmoid contrastive loss"""
        criterion = torch.nn.MultiLabelSoftMarginLoss()
        return criterion(logits, labels.float())

def model_parameter_count(model, model_name='Model'):
    """Count trainable parameters in model"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{model_name} trainable parameters: {total:,}')
    return total
