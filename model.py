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
        
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        
        
        self.lstm = nn.LSTM(
            input_size=16384,
            hidden_size=128,
            batch_first=True
        )
    def forward(self, x):
        
        batch_size, num_frames, channels, height, width = x.shape
        
        x = x.reshape(
            batch_size * num_frames,
            channels,
            height,
            width
        )
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.flatten(start_dim=1)
        
        x.reshape(batch_size, num_frames, -1)
        
        return x