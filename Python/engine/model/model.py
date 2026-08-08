import torch
from torch import nn
from torch.optim import Adam


WIDTH: int = 8
HEIGHT: int = 8
PLAYERS: int = 2
PIECES: int = 6


class ResBlock(nn.Module):
    """Standard Residual Block for Chess Feature Extraction"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out

class ChessBot(nn.Module):
    def __init__(self, num_res_blocks: int = 4, num_channels: int = 128) -> None:
        super().__init__()

        # Initial Stem: Convert 12 input piece channels to internal channel depth (128)
        self.stem = nn.Sequential(
            nn.Conv2d(12, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # Tower of Residual Blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Value Head: Smooth reduction from spatial features to evaluation
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1),  # 1x1 conv compresses channels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),                                # (Batch, 32 * 8 * 8) = 2048
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()                                    # Bounded output [-1.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.value_head(x)
        return x


model = ChessBot()
optimizer = Adam(model.parameters(), 3e-4)
criterion = torch.nn.L1Loss()
