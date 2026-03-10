import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """Triplet loss với online mining"""
    
    def __init__(self, margin=1.5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive, 2)
        distance_negative = F.pairwise_distance(anchor, negative, 2)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class ContrastiveLoss(nn.Module):
    """Contrastive loss cho pairs"""
    
    def __init__(self, margin=1.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embed1, embed2, label):
        # label = 1 nếu cùng lớp, 0 nếu khác lớp
        distance = F.pairwise_distance(embed1, embed2, 2)
        loss = (label) * torch.pow(distance, 2) + \
               (1 - label) * torch.pow(F.relu(self.margin - distance), 2)
        return loss.mean()

class WeightedCrossEntropyLoss(nn.Module):
    """Cross entropy loss với class weights cho imbalanced data"""
    
    def __init__(self, class_counts):
        super(WeightedCrossEntropyLoss, self).__init__()
        # Tính weights: w_j = N / (K * n_j)
        # N: tổng số samples, K: số classes, n_j: số samples class j
        N = sum(class_counts)
        K = len(class_counts)
        weights = torch.tensor([N / (K * count) for count in class_counts])
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        
    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)