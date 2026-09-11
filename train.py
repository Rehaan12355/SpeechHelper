import torch
import torch.nn as nn
from model import LipReadingModel
from torch.utils.data import TensorDataset, DataLoader
from dataset import load_dataset, split_dataset

X, y = load_dataset("data")

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_train = X_train.permute(0, 1, 4, 2, 3)

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

model = LipReadingModel(num_classes=2)

criterion = nn.CrossEntropyLoss()




optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    for clips, labels in train_loader:
        
        optimizer.zero_grad()
        
        outputs = model(clips)
        
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        
        loss.backward()
        
        optimizer.step()
    average_loss = running_loss/len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {average_loss}")