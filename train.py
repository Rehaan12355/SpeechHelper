import torch
import torch.nn as nn
from model import LipReadingModel
from torch.utils.data import TensorDataset, DataLoader
from dataset import load_dataset, split_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


X, y = load_dataset("data")

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

#Have to permute the tensor values because model.py expects the channel to be in the 3rd dimension
X_train = X_train.permute(0, 1, 4, 2, 3)
X_val = X_val.permute(0, 1, 4, 2, 3)
X_test = X_test.permute(0, 1, 4, 2, 3)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False
)

validation_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)
model = LipReadingModel(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()




optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    #Training loop
    for clips, labels in train_loader:
        
        optimizer.zero_grad()
        
        clips = clips.to(device)
        labels = labels.to(device)

        outputs = model(clips)
        
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        
        loss.backward()
        
        optimizer.step()
    average_loss = running_loss/len(train_loader)
    
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    
    #Validation
    with torch.no_grad():
        for clips, labels in validation_loader:
            clips = clips.to(device)
            labels = labels.to(device)            
            
            outputs = model(clips)
            
            loss = criterion(outputs, labels)
            
            validation_loss += loss.item()
            
            predictions = outputs.argmax(dim=1)
            
            correct += (predictions == labels).sum().item()
            
            total += labels.size(0)
    
    
    
    average_validation_loss = validation_loss/len(validation_loader)
    validation_accuracy = correct/total
    
    
    model.train()
    print(
        f"Epoch {epoch + 1}, \n"
        f"Training Loss: {average_loss}, \n"
        f"Validtation loss {average_validation_loss}, \n"
        f"Validation Accuracy {validation_accuracy}"
        )
    
torch.save(model.state_dict(), "lipreading_model.pth")


# Final Test

model.eval()

correct = 0 
total = 0

with torch.no_grad():
    for clips, labels in test_loader:
        clips = clips.to(device)
        labels = labels.to(device)
        
        outputs = model(clips)
        
        predictions = outputs.argmax(dim=1)
        
        correct += (predictions == labels).sum().item()
        
        total += labels.size(0)
    print(
        f"Accuracy={correct/total}"
    )
        