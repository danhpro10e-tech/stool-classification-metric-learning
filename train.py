import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys
from datetime import datetime
import numpy as np

# Thêm đường dẫn để import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.resnet_model import StoolResNet
from models.metric_learning import TripletNet, ContrastiveNet
from utils.data_utils import create_data_loaders, get_class_weights
from utils.loss_functions import WeightedCrossEntropyLoss, TripletLoss, ContrastiveLoss
from utils.metrics import StoolMetrics

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, config, writer):
    model.train()
    metrics = StoolMetrics(num_classes=config.NUM_CLASSES)
    
    loop = tqdm(loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Train]')
    for batch_idx, (images, labels) in enumerate(loop):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        _, predicted = outputs.max(1)
        metrics.update(predicted, labels, loss.item())
        
        # Compute current accuracy for display
        current_metrics = metrics.compute()
        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_metrics["accuracy"]*100:.2f}%'
        })
        
        # Log to TensorBoard
        step = epoch * len(loader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), step)
        writer.add_scalar('Train/Accuracy', current_metrics['accuracy'], step)
    
    return metrics.compute()

def validate(model, loader, criterion, device, epoch, config, writer):
    model.eval()
    metrics = StoolMetrics(num_classes=config.NUM_CLASSES)
    
    loop = tqdm(loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Val]')
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = outputs.max(1)
            metrics.update(predicted, labels, loss.item())
            
            current_metrics = metrics.compute()
            loop.set_postfix({'acc': f'{current_metrics["accuracy"]*100:.2f}%'})
    
    results = metrics.compute()
    
    # Log metrics
    writer.add_scalar('Val/Accuracy', results['accuracy'], epoch)
    writer.add_scalar('Val/MPCA', results['mpca'], epoch)
    writer.add_scalar('Val/MAD', results['mad_overall'], epoch)
    writer.add_scalar('Val/F1', results['f1'], epoch)
    
    # Log per-class accuracy
    for i, acc in enumerate(results['per_class_accuracy']):
        writer.add_scalar(f'Val/Class_{i}_Accuracy', acc, epoch)
    
    return results

def main():
    # Khởi tạo config
    config = Config()
    
    # Kiểm tra đường dẫn dataset
    if not os.path.exists(config.DATA_DIR):
        print(f"Error: Dataset directory {config.DATA_DIR} not found!")
        print(f"Current working directory: {os.getcwd()}")
        print("Please check DATA_DIR in config.py")
        sys.exit(1)
    
    print(f"Using device: {config.DEVICE}")
    print(f"Dataset directory: {config.DATA_DIR}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config.MODEL_NAME}_{timestamp}"
    experiment_dir = os.path.join(config.OUTPUT_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(experiment_dir, 'logs'))
    
    # Save config
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        for key, value in vars(config).items():
            if not key.startswith('__'):
                f.write(f"{key}: {value}\n")
    
    # Create data loaders
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(config)
    
    # Lấy targets từ dataset (xử lý cả OversampledDataset và ImageFolder)
    if hasattr(train_dataset, 'targets'):
        if callable(train_dataset.targets):
            # Nếu targets là property (OversampledDataset)
            train_targets = train_dataset.targets
        else:
            # Nếu targets là attribute (ImageFolder)
            train_targets = train_dataset.targets
    else:
        # Fallback
        train_targets = []
        for _, label in train_dataset:
            train_targets.append(label)
        train_targets = torch.tensor(train_targets)
    
    if hasattr(val_dataset, 'targets'):
        if callable(val_dataset.targets):
            val_targets = val_dataset.targets
        else:
            val_targets = val_dataset.targets
    else:
        val_targets = []
        for _, label in val_dataset:
            val_targets.append(label)
        val_targets = torch.tensor(val_targets)
    
    # Convert to tensor for bincount
    if not isinstance(train_targets, torch.Tensor):
        train_targets = torch.tensor(train_targets)
    if not isinstance(val_targets, torch.Tensor):
        val_targets = torch.tensor(val_targets)
    
    # Print dataset statistics
    train_class_counts = torch.bincount(train_targets, minlength=config.NUM_CLASSES)
    val_class_counts = torch.bincount(val_targets, minlength=config.NUM_CLASSES)
    
    print("\nTraining set distribution:")
    for i in range(config.NUM_CLASSES):
        print(f"  Class {i}: {train_class_counts[i].item()} images")
    
    print("\nValidation set distribution:")
    for i in range(config.NUM_CLASSES):
        print(f"  Class {i}: {val_class_counts[i].item()} images")
    
    # Create model
    print("\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    model = StoolResNet(config)
    model = model.to(config.DEVICE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen layers: {config.FREEZE_PERCENTAGE*100:.0f}%")
    
    # Loss function
    if config.USE_CLASS_WEIGHTS:
        # Sử dụng class weights từ distribution
        from utils.data_utils import get_class_weights
        class_weights = get_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
        print(f"\nUsing weighted loss with class weights")
        print(f"Class weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        print(f"\nUsing standard CrossEntropyLoss")
    
    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler - ❌ ĐÃ XÓA tham số verbose
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=1e-6
        # verbose=True  <- ĐÃ XÓA dòng này
    )
    
    # Training loop
    best_val_acc = 0.0
    best_val_mad = float('inf')
    patience_counter = 0
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    for epoch in range(config.EPOCHS):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            config.DEVICE, epoch, config, writer
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, 
            config.DEVICE, epoch, config, writer
        )
        
        # Update scheduler
        scheduler.step(val_metrics['accuracy'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # In thông báo thủ công khi LR thay đổi
        if epoch > 0 and hasattr(scheduler, '_last_lr') and scheduler._last_lr != current_lr:
            print(f"  → Learning rate reduced to {current_lr:.6f}")
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.EPOCHS} Summary:")
        print(f"  Train Loss: {train_metrics['mean_loss']:.4f} | Train Acc: {train_metrics['accuracy']*100:.2f}%")
        print(f"  Val Loss: {val_metrics['mean_loss']:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}%")
        print(f"  Val MPCA: {val_metrics['mpca']*100:.2f}% | Val MAD: {val_metrics['mad_overall']:.3f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model (based on accuracy)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_val_mad = val_metrics['mad_overall']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_mpca': val_metrics['mpca'],
                'val_mad': best_val_mad,
                'config': vars(config)
            }
            torch.save(checkpoint, os.path.join(experiment_dir, 'best_model.pth'))
            print(f"  ✓ New best model saved! (Acc: {best_val_acc*100:.2f}%, MAD: {best_val_mad:.3f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(experiment_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(experiment_dir, 'final_model.pth'))
    writer.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Best validation MAD: {best_val_mad:.3f}")
    print(f"Results saved to: {experiment_dir}")
    print("="*60)

if __name__ == '__main__':
    main()