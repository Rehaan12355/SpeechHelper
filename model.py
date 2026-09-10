import torch
import torch.nn as nn


class LipReadingModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
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

        self.fc = nn.Linear(
            in_features=128,
            out_features=num_classes
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

        x = x.reshape(batch_size, num_frames, -1)

        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = LipReadingModel(num_classes=5)

    x = torch.randn(8, 45, 1, 64, 128)

    output = model(x)

    print(output.shape)
