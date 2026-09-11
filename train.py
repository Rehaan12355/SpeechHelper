import torch
import torch.nn as nn
from model import LipReadingModel
from torch.utils.data import DataLoader


model = LipReadingModel(num_classes=5)

criterion = nn.CrossEntropyLoss()

num_epochs = 0


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(num_epochs):
    
    for clips, labels in train_loader:
        
        optimizer.zero_grad()
        
        outputs = model(clips)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()