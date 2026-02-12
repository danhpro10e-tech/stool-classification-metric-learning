import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Transform =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===== Load model (ResNet18 embedding) =====
model = models.resnet18(pretrained=True)
model.fc = nn.Identity()   # lấy embedding 512-d
model = model.to(device)
model.eval()

# ===== Dataset =====
train_ds = ImageFolder("../dataset/train", transform=transform)
val_ds   = ImageFolder("../dataset/val", transform=transform)
test_ds  = ImageFolder("../dataset/test", transform=transform)

def extract_embeddings(dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    embs, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            e = model(x)
            embs.append(e.cpu())
            labels.append(y)

    return torch.cat(embs), torch.cat(labels)

# ===== Extract =====
X_train, y_train = extract_embeddings(train_ds)
X_test, y_test   = extract_embeddings(test_ds)

# Convert sang numpy cho sklearn
X_train = X_train.numpy()
y_train = y_train.numpy()
X_test  = X_test.numpy()
y_test  = y_test.numpy()

# ===== kNN =====
knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn.fit(X_train, y_train)

acc = knn.score(X_test, y_test)
print("kNN accuracy:", acc)
