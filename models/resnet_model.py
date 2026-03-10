import torch
import torch.nn as nn
import torchvision.models as models

class StoolResNet(nn.Module):
    """ResNet model cho phân loại ảnh phân với layer freezing tối ưu"""
    
    def __init__(self, config):
        super(StoolResNet, self).__init__()
        self.config = config
        self.num_classes = config.NUM_CLASSES
        
        # Load pretrained model
        if config.MODEL_NAME == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if config.PRETRAINED else None
            self.backbone = models.resnet18(weights=weights)
            self.feature_dim = 512
        elif config.MODEL_NAME == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if config.PRETRAINED else None
            self.backbone = models.resnet34(weights=weights)
            self.feature_dim = 512
        elif config.MODEL_NAME == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if config.PRETRAINED else None
            self.backbone = models.resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {config.MODEL_NAME}")
        
        # Freeze layers theo khuyến nghị (70% layers đầu)
        self._freeze_layers(config.FREEZE_PERCENTAGE)
        
        # Thay thế classifier head
        self.backbone.fc = nn.Identity()  # Bỏ classifier gốc để lấy features
        
        # Classifier cho classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )
        
        # Embedding head cho metric learning (nếu cần)
        self.embedding_head = nn.Sequential(
            nn.Linear(self.feature_dim, config.EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(config.EMBEDDING_DIM, config.EMBEDDING_DIM)
        )
        
    def _freeze_layers(self, freeze_percentage):
        """Đông cứng các layers đầu theo tỷ lệ phần trăm"""
        # Đếm tổng số layers có thể train
        trainable_layers = []
        for name, module in self.backbone.named_children():
            if name not in ['fc']:  # Không tính fc layer
                if hasattr(module, 'parameters'):
                    trainable_layers.append(name)
        
        num_layers_to_freeze = int(len(trainable_layers) * freeze_percentage)
        layers_to_freeze = trainable_layers[:num_layers_to_freeze]
        
        print(f"Freezing {num_layers_to_freeze}/{len(trainable_layers)} layers: {layers_to_freeze}")
        
        # Đông cứng các layers được chọn
        for name, param in self.backbone.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False
    
    def forward(self, x, return_embedding=False):
        # Extract features từ backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        
        # Classification
        logits = self.classifier(features)
        
        if return_embedding:
            embeddings = self.embedding_head(features)
            return logits, embeddings
        
        return logits
    
    def get_features(self, x):
        """Lấy feature embeddings cho metric learning"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        
        return self.embedding_head(features)