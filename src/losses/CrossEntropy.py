from __future__ import annotations

import torch
import torch.nn as nn


class CrossEntropy(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss
