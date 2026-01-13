import os, random, torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class TripletDataset(Dataset):
    def __init__(self, root):
        self.ds = datasets.ImageFolder(root)
        self.map = {}
        for i, (_, y) in enumerate(self.ds.samples):
            self.map.setdefault(y, []).append(i)

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        a_path, y = self.ds.samples[idx]
        p_path, _ = self.ds.samples[random.choice(self.map[y])]
        n_label = random.choice([k for k in self.map if k != y])
        n_path, _ = self.ds.samples[random.choice(self.map[n_label])]

        def load(p): return transform(Image.open(p).convert("RGB"))
        return load(a_path), load(p_path), load(n_path)

class ResNetEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, dim)

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        return nn.functional.normalize(self.fc(x), p=2, dim=1)

train_ds = TripletDataset("../dataset/train")
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

model = ResNetEmbedding().to(DEVICE)
loss_fn = nn.TripletMarginLoss(margin=1.0)
opt = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(20):
    model.train()
    total = 0
    for a,p,n in train_loader:
        a,p,n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(a), model(p), model(n))
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch+1} | Triplet Loss: {total:.3f}")

os.makedirs("../outputs", exist_ok=True)
torch.save(model.state_dict(), "../outputs/triplet_model.pth")
