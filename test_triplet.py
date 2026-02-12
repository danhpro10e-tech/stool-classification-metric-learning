import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===== Model Definition =====
class ResNetEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        base = models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, dim)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        return nn.functional.normalize(self.fc(x), p=2, dim=1)

# Load model
model = ResNetEmbedding()
model.load_state_dict(torch.load("../outputs/triplet_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load datasets
train_ds = datasets.ImageFolder("../dataset/train", transform=transform)
val_ds   = datasets.ImageFolder("../dataset/val", transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

def extract_embeddings(loader):
    embs = []
    labels = []

    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(DEVICE)
            e = model(imgs)
            embs.append(e.cpu())
            labels.append(y)

    return torch.cat(embs), torch.cat(labels)

# Extract
X_train, y_train = extract_embeddings(train_loader)
X_val, y_val     = extract_embeddings(val_loader)

X_train = X_train.numpy()
y_train = y_train.numpy()
X_val   = X_val.numpy()
y_val   = y_val.numpy()

# kNN
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X_train, y_train)

acc = knn.score(X_val, y_val)
print(f"\nTriplet + kNN Accuracy: {acc*100:.2f}%")

