import torch
from torchvision import datasets
from train_triplet import ResNetEmbedding
from utils import build_prototypes
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

model = ResNetEmbedding().to(DEVICE)
model.load_state_dict(torch.load("../outputs/triplet_model.pth"))
model.eval()

train_ds = datasets.ImageFolder("../dataset/train")
protos = build_prototypes(model, train_ds, DEVICE)

img = tf(Image.open("test_stool.jpg").convert("RGB")).unsqueeze(0).to(DEVICE)
emb = model(img)

dists = {k: torch.norm(emb - v).item() for k,v in protos.items()}
pred = min(dists, key=dists.get)

print("Predicted Bristol Stool Type:", pred + 1)
