import torch
import torch.nn as nn


class GalaxyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self._example_input_array = torch.randn((1, 3, 77, 77))

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=6),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4),
            nn.ReLU(),
        )

        n_channels = (
            self.conv4(
                self.conv3(self.conv2(self.conv1(self._example_input_array)))
            )
            .view(-1)
            .shape[0]
        )

        self.fc1 = nn.Linear(n_channels, 120)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(120, 80)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(80, 60)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(60, 37)
        self.dropout4 = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = torch.flatten(x, 1)

        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.dropout4(x)

        output = torch.sigmoid(x)
        return output


