from __future__ import annotations

import torch
from torch import nn


class MNISTClassifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        self.lr_rate = 0.001

    def forward(self, x):
        #   batch_size, channels, width, height = x.size()

        #   # (b, 1, 28, 28) -> (b, 1*28*28)
        #   x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x.float())
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        return x
