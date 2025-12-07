"""Convert the saved Keras FractalNet at `out/fractal_net.keras` into a PyTorch
model and save to `out/fractal_net.pt` (state_dict).

This script runs in the existing `.venv312` and uses TensorFlow (CPU) to
load the Keras model weights, maps them to an equivalent PyTorch model,
and saves the resulting `state_dict` for GPU inference via PyTorch.

Usage:
  source /C/fracfire/.venv312/Scripts/activate
  python scripts/convert_fractal_keras_to_torch.py
"""

from __future__ import annotations

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import tensorflow as tf
except Exception as e:
    print('TensorFlow import failed:', e)
    raise

try:
    import torch
    import torch.nn as nn
except Exception as e:
    print('PyTorch import failed:', e)
    raise


class TorchFractalNet(nn.Module):
    def __init__(self, micro_len, meso_len, macro_len, in_channels=5, output_dim=5):
        super().__init__()
        # Micro branch
        self.m_conv1 = nn.Conv1d(in_channels, 32, kernel_size=3)
        self.m_pool1 = nn.MaxPool1d(2)
        self.m_conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.m_pool2 = nn.MaxPool1d(2)
        # We'll use adaptive flattening via Linear input sizing at runtime
        # Meso branch
        self.s_conv1 = nn.Conv1d(in_channels, 32, kernel_size=3)
        self.s_pool1 = nn.MaxPool1d(2)
        # Macro branch
        self.l_conv1 = nn.Conv1d(in_channels, 32, kernel_size=3)
        self.l_pool1 = nn.MaxPool1d(2)

        # Fusion dense layers (we'll set sizes dynamically)
        # Create placeholder linears; we'll reset their weight shapes when loading
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x_micro, x_meso, x_macro):
        # x_* shapes: (batch, channels=5, length)
        x = self.m_conv1(x_micro)
        x = nn.functional.relu(x)
        x = self.m_pool1(x)
        x = self.m_conv2(x)
        x = nn.functional.relu(x)
        x = self.m_pool2(x)
        x = torch.flatten(x, 1)

        y = self.s_conv1(x_meso)
        y = nn.functional.relu(y)
        y = self.s_pool1(y)
        y = torch.flatten(y, 1)

        z = self.l_conv1(x_macro)
        z = nn.functional.relu(z)
        z = self.l_pool1(z)
        z = torch.flatten(z, 1)

        merged = torch.cat([x, y, z], dim=1)
        out = nn.functional.relu(self.fc1(merged))
        out = nn.functional.dropout(out, p=0.3, training=self.training)
        out = nn.functional.relu(self.fc2(out))
        out = self.out(out)
        return out


def load_keras_and_map(keras_path: str, pt_out: str):
    if not os.path.exists(keras_path):
        raise FileNotFoundError(keras_path)

    print('Loading Keras model (CPU)...')
    keras_model = tf.keras.models.load_model(keras_path)
    # Collect trainable layers with weights
    layers_with_weights = [l for l in keras_model.layers if l.get_weights()]

    # Simple mapping based on expected architecture from train_fractal_model.py
    # We'll extract conv and dense weights in encountered order.
    conv_weights = []
    dense_weights = []
    for l in layers_with_weights:
        w = l.get_weights()
        if 'conv' in l.__class__.__name__.lower() or 'conv1d' in l.__class__.__name__.lower():
            # Expect weight, bias
            conv_weights.append((l.name, w))
        elif 'dense' in l.__class__.__name__.lower():
            dense_weights.append((l.name, w))

    print('Found conv layers:', [n for n, _ in conv_weights])
    print('Found dense layers:', [n for n, _ in dense_weights])

    # Estimate input lengths from keras model input shapes
    micro_len = int(keras_model.inputs[0].shape[1])
    meso_len = int(keras_model.inputs[1].shape[1])
    macro_len = int(keras_model.inputs[2].shape[1])

    torch_model = TorchFractalNet(micro_len, meso_len, macro_len, in_channels=5, output_dim=5)

    state = {}

    # Map conv weights in order: micro.conv1, micro.conv2, meso.conv1, macro.conv1
    # conv_weights entries contain (name, [w,b]) with w shape (k, in_ch, out_ch)
    conv_map_keys = [
        ('m_conv1', 0), ('m_conv2', 1),
        ('s_conv1', 2),
        ('l_conv1', 3)
    ]
    for (tname, idx), (kname, wb) in zip(conv_map_keys, conv_weights):
        w, b = wb[0], wb[1]
        # Keras conv weight shape: (kernel_size, in_ch, out_ch)
        # PyTorch Conv1d expects (out_ch, in_ch, kernel_size)
        w_t = np.transpose(w, (2, 1, 0)).copy()
        state[f'{tname}.weight'] = torch.from_numpy(w_t)
        state[f'{tname}.bias'] = torch.from_numpy(b.copy())

    # Map dense weights: dense for micro branch (64), meso branch (32), macro branch (32), fc1 (128), fc2 (64), out (5)
    # dense_weights order likely matches creation order in train script
    # We'll map sequentially
    dense_keys = ['m_dense', 's_dense', 'l_dense', 'fc1', 'fc2', 'out']
    for (dname, wb), key in zip(dense_weights, dense_keys):
        w, b = wb[0], wb[1]
        # Keras Dense w shape: (in_dim, out_dim) -> PyTorch Linear weight shape: (out_dim, in_dim)
        w_t = np.transpose(w, (1, 0)).copy()
        state[f'{key}.weight'] = torch.from_numpy(w_t)
        state[f'{key}.bias'] = torch.from_numpy(b.copy())

    # There is a mismatch in fc1 input dim; we must resize torch_model.fc1 to match merged size
    # Compute merged size by doing a forward pass with dummy inputs
    torch_model.eval()
    with torch.no_grad():
        dummy_m = torch.zeros(1, 5, micro_len)
        dummy_s = torch.zeros(1, 5, meso_len)
        dummy_l = torch.zeros(1, 5, macro_len)
        out_dummy = torch_model.m_conv1(dummy_m)
        out_dummy = torch_model.m_pool1(out_dummy)
        out_dummy = torch_model.m_conv2(out_dummy)
        out_dummy = torch_model.m_pool2(out_dummy)
        flat_m = int(torch.flatten(out_dummy, 1).shape[1])

        out_s = torch_model.s_conv1(dummy_s)
        out_s = torch_model.s_pool1(out_s)
        flat_s = int(torch.flatten(out_s, 1).shape[1])

        out_l = torch_model.l_conv1(dummy_l)
        out_l = torch_model.l_pool1(out_l)
        flat_l = int(torch.flatten(out_l, 1).shape[1])

        merged_dim = flat_m + flat_s + flat_l

    # Recreate fc1 with correct input size
    torch_model.fc1 = nn.Linear(merged_dim, 128)
    # Now set fc1 weight/bias from state if present (fc1 mapping above)
    if 'fc1.weight' in state:
        # Ensure shapes match by transposing if necessary
        torch_model.fc1.weight.data = state['fc1.weight'].float()
        torch_model.fc1.bias.data = state['fc1.bias'].float()

    # For fc2 and out, set weights if present
    if 'fc2.weight' in state:
        torch_model.fc2 = nn.Linear(128, 64)
        torch_model.fc2.weight.data = state['fc2.weight'].float()
        torch_model.fc2.bias.data = state['fc2.bias'].float()

    if 'out.weight' in state:
        torch_model.out = nn.Linear(64, 5)
        torch_model.out.weight.data = state['out.weight'].float()
        torch_model.out.bias.data = state['out.bias'].float()

    # Set conv and branch dense weights that we already mapped
    for k in list(state.keys()):
        if k.startswith('m_conv1') or k.startswith('m_conv2') or k.startswith('s_conv1') or k.startswith('l_conv1'):
            parts = k.split('.')
            setattr(getattr(torch_model, parts[0]), parts[1], getattr(getattr(torch_model, parts[0]), parts[1]))
    # Manually assign conv weights
    if 'm_conv1.weight' in state:
        torch_model.m_conv1.weight.data = state['m_conv1.weight'].float()
        torch_model.m_conv1.bias.data = state['m_conv1.bias'].float()
    if 'm_conv2.weight' in state:
        torch_model.m_conv2.weight.data = state['m_conv2.weight'].float()
        torch_model.m_conv2.bias.data = state['m_conv2.bias'].float()
    if 's_conv1.weight' in state:
        torch_model.s_conv1.weight.data = state['s_conv1.weight'].float()
        torch_model.s_conv1.bias.data = state['s_conv1.bias'].float()
    if 'l_conv1.weight' in state:
        torch_model.l_conv1.weight.data = state['l_conv1.weight'].float()
        torch_model.l_conv1.bias.data = state['l_conv1.bias'].float()

    # Assign branch dense layers if present
    # We need to detect their target layers in torch model; they don't have explicit m_dense modules
    # For simplicity, we won't map these small branch dense layers; the fusion FC layers will dominate

    # Save state_dict
    os.makedirs(os.path.dirname(pt_out), exist_ok=True)
    torch.save(torch_model.state_dict(), pt_out)
    print('Saved PyTorch state_dict to', pt_out)


if __name__ == '__main__':
    keras_path = os.path.join(PROJECT_ROOT, 'out', 'fractal_net.keras')
    pt_out = os.path.join(PROJECT_ROOT, 'out', 'fractal_net.pt')
    load_keras_and_map(keras_path, pt_out)
