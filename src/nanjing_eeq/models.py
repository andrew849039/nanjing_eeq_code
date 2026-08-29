from __future__ import annotations

import torch
from torch import nn


def center_slice(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        h = tensor.shape[-2] // 2
        w = tensor.shape[-1] // 2
        return tensor[..., h, w]
    if tensor.ndim == 3:
        h = tensor.shape[-2] // 2
        w = tensor.shape[-1] // 2
        return tensor[:, h, w]
    return tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetBackbone(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.up3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up1 = UpBlock(base_channels * 2, base_channels, base_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        x = self.bottleneck(self.pool(s3))
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return x


class PlainUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.backbone = UNetBackbone(in_channels, base_channels)
        self.score_head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        score_map = torch.sigmoid(self.score_head(feat)).squeeze(1)
        return {"score_map": score_map}


class PatchMLP(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = in_channels * patch_size * patch_size
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(p=0.1)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        score = torch.sigmoid(self.net(x.flatten(start_dim=1))).squeeze(1)
        return {"score_map": score}
