# Code Dump: 06_ml_models

## File: src/ml/models/tilt.py
```python
"""
Neural Tilt Model

A lightweight PyTorch model that outputs a "tilt" vector to adjust the 
probabilities of the PriceGenerator's next move.

It does NOT predict the next price directly. It predicts the *bias* 
of the next tick (up/down probability, volatility scaling).
"""

import torch
import torch.nn as nn
from typing import Dict, Any

class TiltModel(nn.Module):
    """
    Neural network for estimating market tilt.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Simple MLP for now, can be GRU/LSTM later
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Outputs: [up_prob_bias, vol_bias, persistence_bias]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Feature tensor (batch, input_dim)
            
        Returns:
            Tilt vector (batch, 3)
        """
        return self.net(x)
    
    def save(self, path: str):
        """Save model weights"""
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        """Load model weights"""
        self.load_state_dict(torch.load(path))

```

---

## File: src/ml/models/cnn.py
```python
"""
Convolutional Neural Network (CNN) model for price pattern recognition.
"""
import torch.nn as nn

class PriceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Implement CNN architecture
        pass

```

---

## File: src/ml/models/generative.py
```python
"""
Generative model for synthetic price action.
"""
import torch.nn as nn

class GenerativePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Implement generative model architecture
        pass

```

---

