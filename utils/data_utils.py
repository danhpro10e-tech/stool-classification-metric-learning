import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
import numpy as np
from PIL import Image

# ❌ XÓA dòng này vì nó gây circular import
# from utils.data_utils import create_data_loaders, get_class_weights

class OversampledDataset(Dataset):
    """Dataset với oversampling cho các lớp hiếm"""
    
    def __init__(self, base_dataset, oversampling_factors):
        self.base_dataset = base_dataset
        self.oversampling_factors = oversampling_factors
        self.classes = base_dataset.classes
        
        # Tạo indices với oversampling
        self.indices = []
        class_indices = [[] for _ in range(len(base_dataset.classes))]
        
        for idx, (_, label) in enumerate(base_dataset):
            class_indices[label].append(idx)
        
        for class_id, factor in oversampling_factors.items():
            # Nhân bản các samples của lớp hiếm
            if class_id < len(class_indices):  # Kiểm tra class_id hợp lệ
                original_indices = class_indices[class_id]
                for _ in range(factor):
                    self.indices.extend(original_indices)
        
        # Thêm các lớp khác (không oversampling)
        for class_id in range(len(base_dataset.classes)):
            if class_id not in oversampling_factors:
                self.indices.extend(class_indices[class_id])
    
    @property
    def targets(self):
        """Lấy targets từ dataset gốc thông qua indices"""
        return [self.base_dataset.targets[idx] for idx in self.indices]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

def get_class_weights(dataset):
    """Tính class weights cho weighted loss"""
    if hasattr(dataset, 'targets'):
        if callable(dataset.targets):
            # Nếu targets là property (OversampledDataset)
            targets = dataset.targets
        else:
            # Nếu targets là attribute (ImageFolder)
            targets = dataset.targets
    else:
        # Fallback: collect targets manually
        targets = []
        for _, label in dataset:
            targets.append(label)
        targets = torch.tensor(targets)
    
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    
    class_counts = torch.bincount(targets)
    total = len(targets)
    weights = total / (len(class_counts) * class_counts.float())
    return weights

def create_data_loaders(config):
    """Tạo train/val data loaders với augmentation và oversampling"""
    
    # Transform cho training (với augmentation)
    train_transform_list = [
        transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
    ]
    
    if hasattr(config, 'RANDOM_HORIZONTAL_FLIP') and config.RANDOM_HORIZONTAL_FLIP:
        train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    if hasattr(config, 'RANDOM_VERTICAL_FLIP') and config.RANDOM_VERTICAL_FLIP:
        train_transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    
    if hasattr(config, 'RANDOM_ROTATION') and config.RANDOM_ROTATION:
        train_transform_list.append(transforms.RandomRotation(config.RANDOM_ROTATION))
    
    if hasattr(config, 'COLOR_JITTER') and config.COLOR_JITTER:
        train_transform_list.append(
            transforms.ColorJitter(**config.COLOR_JITTER)
        )
    
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose(train_transform_list)
    
    # Transform cho validation (không augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_path = os.path.join(config.DATA_DIR, 'train')
    val_path = os.path.join(config.DATA_DIR, 'val')
    
    # Kiểm tra đường dẫn
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training path not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation path not found: {val_path}")
    
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    
    print(f"Training samples (original): {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Oversampling cho training
    if hasattr(config, 'OVERSAMPLING') and config.OVERSAMPLING:
        train_dataset = OversampledDataset(train_dataset, config.OVERSAMPLING_FACTORS)
        print(f"Training samples (after oversampling): {len(train_dataset)}")
    
    # Tạo data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Tránh lỗi multiprocessing trên Windows
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    return train_loader, val_loader, train_dataset, val_dataset