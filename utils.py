import torch
from PIL import Image
from torchvision import transforms

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@torch.no_grad()
def build_prototypes(model, dataset, device):
    model.eval()
    protos = {}
    for path, y in dataset.samples:
        img = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        emb = model(img)
        protos.setdefault(y, []).append(emb)
    return {k: torch.mean(torch.cat(v),0) for k,v in protos.items()}
