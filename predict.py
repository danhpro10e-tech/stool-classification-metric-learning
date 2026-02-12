import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Transform (giống lúc train) =====
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ===== Load test dataset =====
test_ds = datasets.ImageFolder("../dataset/val", transform=transform)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

class_names = test_ds.classes

# ===== Load model =====
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("../outputs/model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ===== Evaluate =====
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print("Total samples:", total)
print("Correct predictions:", correct)
print("Test Accuracy: {:.2f}%".format(accuracy))
