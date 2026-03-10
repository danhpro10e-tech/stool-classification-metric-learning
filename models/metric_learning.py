import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_model import StoolResNet

class TripletNet(nn.Module):
    """Wrapper cho triplet loss training"""
    
    def __init__(self, config):
        super(TripletNet, self).__init__()
        self.config = config
        self.backbone = StoolResNet(config)
        
    def forward(self, anchor, positive, negative):
        """Forward pass cho triplet"""
        anchor_embed = self.backbone.get_features(anchor)
        positive_embed = self.backbone.get_features(positive)
        negative_embed = self.backbone.get_features(negative)
        
        return anchor_embed, positive_embed, negative_embed
    
    def get_embedding(self, x):
        return self.backbone.get_features(x)

class ContrastiveNet(nn.Module):
    """Wrapper cho contrastive loss training"""
    
    def __init__(self, config):
        super(ContrastiveNet, self).__init__()
        self.config = config
        self.backbone = StoolResNet(config)
        
    def forward(self, x1, x2):
        embed1 = self.backbone.get_features(x1)
        embed2 = self.backbone.get_features(x2)
        return embed1, embed2
    
    def get_embedding(self, x):
        return self.backbone.get_features(x)