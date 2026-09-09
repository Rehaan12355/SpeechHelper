import torch 
import torch.nn as nn


class LipReadingModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernerl_size=3,
            padding=1
        )
        
        self.relu = nn.ReLU()
        
        self.pool = nn.MaxPool2d(
            kernel_size=2
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x