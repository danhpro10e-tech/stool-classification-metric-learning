import torch

class Config:
    # Data paths
    DATA_DIR = "../dataset"
    OUTPUT_DIR = "../outputs"
    
    # Model parameters
    MODEL_NAME = 'resnet18'  # resnet18, resnet34, resnet50
    NUM_CLASSES = 7
    IMAGE_SIZE = 224
    PRETRAINED = True
    FREEZE_PERCENTAGE = 0.7  # Đông cứng 70% layers theo khuyến nghị
    
    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    
    # Loss function weights
    USE_CLASS_WEIGHTS = True  # Cân bằng loss cho các lớp hiếm
    
    # Data augmentation
    USE_AUGMENTATION = True
    RANDOM_HORIZONTAL_FLIP = True
    RANDOM_VERTICAL_FLIP = True
    RANDOM_ROTATION = 15
    COLOR_JITTER = {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.1,
        'hue': 0.1
    }
    
    # Oversampling cho các lớp hiếm
    OVERSAMPLING = True
    OVERSAMPLING_FACTORS = {
        0: 5,  # Type-1: nhân 5 lần
        1: 2,  # Type-2: nhân 2 lần
        5: 2,  # Type-6: nhân 2 lần
        6: 2,  # Type-7: nhân 2 lần
    }
    
    # Metric learning (tùy chọn, không bắt buộc)
    USE_METRIC_LEARNING = False
    TRIPLET_MARGIN = 1.5
    CONTRASTIVE_MARGIN = 1.5
    EMBEDDING_DIM = 128
    
    # Training optimization
    EARLY_STOPPING_PATIENCE = 10
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Evaluation metrics
    TRACK_ABSOLUTE_DEVIATION = True  # Theo dõi độ lệch tuyệt đối (quan trọng)