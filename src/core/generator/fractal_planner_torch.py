"""Lightweight PyTorch implementation of the FractalNet architecture.

This module provides `TorchFractalNet` which mirrors the Keras model
used for FractalPlanner so we can run inference with PyTorch on GPU
from the existing `.venv312` (which already has GPU-capable PyTorch).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TorchFractalNet(nn.Module):
    def __init__(self, micro_len: int, meso_len: int, macro_len: int, in_channels: int = 5, output_dim: int = 5):
        super().__init__()
        # Micro branch
        self.m_conv1 = nn.Conv1d(in_channels, 32, kernel_size=3)
        self.m_pool1 = nn.MaxPool1d(2)
        self.m_conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.m_pool2 = nn.MaxPool1d(2)

        # Meso branch
        self.s_conv1 = nn.Conv1d(in_channels, 32, kernel_size=3)
        self.s_pool1 = nn.MaxPool1d(2)

        # Macro branch
        self.l_conv1 = nn.Conv1d(in_channels, 32, kernel_size=3)
        self.l_pool1 = nn.MaxPool1d(2)

        # We'll compute flattened sizes dynamically using example forward
        # Create fusion MLP with placeholder input size; will be reset if needed
        self.fc1 = nn.Linear(1, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)

    def _feature_size(self, length: int, conv1, pool1, conv2=None, pool2=None) -> int:
        # Create a dummy tensor and run through layers to compute flatten size
        x = torch.zeros(1, 5, length)
        x = F.relu(conv1(x))
        x = pool1(x)
        if conv2 is not None:
            x = F.relu(conv2(x))
        if pool2 is not None:
            x = pool2(x)
        return int(torch.flatten(x, 1).shape[1])

    def reset_fc(self, micro_len: int, meso_len: int, macro_len: int):
        fm = self._feature_size(micro_len, self.m_conv1, self.m_pool1, self.m_conv2, self.m_pool2)
        fs = self._feature_size(meso_len, self.s_conv1, self.s_pool1)
        fl = self._feature_size(macro_len, self.l_conv1, self.l_pool1)
        merged = fm + fs + fl
        self.fc1 = nn.Linear(merged, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, self.out.out_features if hasattr(self.out, 'out_features') else 5)

    def forward(self, x_micro, x_meso, x_macro):
        # Expect inputs shape: (batch, length, channels) OR (batch, channels, length)
        # We'll accept (batch, length, channels) and transpose if needed
        if x_micro.ndim == 3 and x_micro.shape[1] != 5:
            # assume (batch, length, channels)
            x_micro = x_micro.permute(0, 2, 1)
        if x_meso.ndim == 3 and x_meso.shape[1] != 5:
            x_meso = x_meso.permute(0, 2, 1)
        if x_macro.ndim == 3 and x_macro.shape[1] != 5:
            x_macro = x_macro.permute(0, 2, 1)

        xm = F.relu(self.m_conv1(x_micro))
        xm = self.m_pool1(xm)
        xm = F.relu(self.m_conv2(xm))
        xm = self.m_pool2(xm)
        xm = torch.flatten(xm, 1)

        xs = F.relu(self.s_conv1(x_meso))
        xs = self.s_pool1(xs)
        xs = torch.flatten(xs, 1)

        xl = F.relu(self.l_conv1(x_macro))
        xl = self.l_pool1(xl)
        xl = torch.flatten(xl, 1)

        merged = torch.cat([xm, xs, xl], dim=1)
        z = F.relu(self.fc1(merged))
        z = self.dropout(z)
        z = F.relu(self.fc2(z))
        out = self.out(z)
        return out


def build_torch_model_from_shapes(micro_len: int, meso_len: int, macro_len: int) -> Tuple[TorchFractalNet, int]:
    net = TorchFractalNet(micro_len, meso_len, macro_len)
    net.reset_fc(micro_len, meso_len, macro_len)
    return net, net.out.out_features
