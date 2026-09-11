import torch

from model import LipReadingModel
from preprocess import preprocess_clip


LABELS = {
    0: "hello",
    1: "yes"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LipReadingModel(num_classes=2).to(device)

model.load_state_dict(
    torch.load("lipreading_model.pth", map_location=device)
)

model.eval()
clip = preprocess_clip("data/hello/clip_101")

clip = torch.tensor(
    clip,
    dtype=torch.float32
)

clip = clip.permute(0, 3, 1, 2)

clip = clip.unsqueeze(0)

clip = clip.to(device)

with torch.no_grad():
    output = model(clip)
    prediction = output.argmax(dim=1).item()
    
    print(LABELS[prediction])
    
