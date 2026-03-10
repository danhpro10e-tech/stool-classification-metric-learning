import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import argparse

from config import Config
from models.resnet_model import StoolResNet
from utils.data_utils import create_data_loaders
from utils.metrics import StoolMetrics

def evaluate_model(model_path, config):
    """Evaluate a trained model on validation set"""
    
    # Load model
    model = StoolResNet(config)
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint.get('val_acc', 0)
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} with val_acc: {best_acc*100:.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(config.DEVICE)
    model.eval()
    
    # Create data loaders
    _, val_loader, _, val_dataset = create_data_loaders(config)
    
    # Evaluate
    criterion = nn.CrossEntropyLoss()
    metrics = StoolMetrics(num_classes=config.NUM_CLASSES)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            metrics.update(predicted, labels, loss.item())
    
    # Print results
    results = metrics.print_summary()
    
    # Per-class analysis
    print("\nPer-Class Detailed Analysis:")
    print("-" * 40)
    class_counts = torch.bincount(torch.tensor(val_dataset.targets))
    for i in range(config.NUM_CLASSES):
        print(f"Class {i} ({val_dataset.classes[i]}): {class_counts[i]} samples")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    args = parser.parse_args()
    
    config = Config()
    evaluate_model(args.model_path, config)