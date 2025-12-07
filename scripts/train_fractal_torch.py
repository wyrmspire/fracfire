"""Train FractalNet in PyTorch on GPU.

Usage:
  source /C/fracfire/.venv312/Scripts/activate
  python scripts/train_fractal_torch.py --epochs 3 --batch-size 64 --out out/fractal_net_torch.pt

This mirrors `scripts/train_fractal_model.py` but uses PyTorch and runs on GPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.generator.fractal_planner_torch import build_torch_model_from_shapes


def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    X_micro = data['X_micro']
    X_meso = data['X_meso']
    X_macro = data['X_macro']
    Y = data['Y_targets']
    return X_micro, X_meso, X_macro, Y


def to_tensor_dataset(Xm, Xs, Xl, Y):
    # Convert numpy arrays to torch tensors
    # Expect shapes: (N, L, C)
    Xm_t = torch.from_numpy(Xm.astype('float32'))
    Xs_t = torch.from_numpy(Xs.astype('float32'))
    Xl_t = torch.from_numpy(Xl.astype('float32'))
    Y_t = torch.from_numpy(Y.astype('float32'))
    ds = TensorDataset(Xm_t, Xs_t, Xl_t, Y_t)
    return ds


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    X_micro, X_meso, X_macro, Y = load_dataset(os.path.join(PROJECT_ROOT, 'data', 'processed', 'fractal_dataset.npz'))
    print('Loaded dataset. Samples=', len(Y))

    ds = to_tensor_dataset(X_micro, X_meso, X_macro, Y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    micro_len = X_micro.shape[1]
    meso_len = X_meso.shape[1]
    macro_len = X_macro.shape[1]

    model, _ = build_torch_model_from_shapes(micro_len, meso_len, macro_len)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for i, (xm, xs, xl, y) in enumerate(loader):
            xm = xm.to(device)
            xs = xs.to(device)
            xl = xl.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(xm, xs, xl)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f'Epoch {epoch+1} Batch {i+1}/{len(loader)} Loss {(running_loss / (i+1)):.6f}')

        t1 = time.time()
        print(f'Epoch {epoch+1} finished. Avg loss={(running_loss/len(loader)):.6f}. Time={(t1-t0):.2f}s')

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print('Saved PyTorch model state to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out', type=str, default=os.path.join(PROJECT_ROOT, 'out', 'fractal_net_torch.pt'))
    args = parser.parse_args()
    train(args)
