"""
Models package for stool classification.
Contains ResNet models and metric learning implementations.
"""

from .resnet_model import StoolResNet
from .metric_learning import TripletNet, ContrastiveNet

__all__ = [
    'StoolResNet',
    'TripletNet',
    'ContrastiveNet'
]