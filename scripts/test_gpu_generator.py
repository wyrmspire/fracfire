import time
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(ROOT, os.pardir))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from src.core.generator.engine import PriceGenerator, PhysicsConfig


def main():
    gen = PriceGenerator(physics_config=PhysicsConfig())
    try:
        import torch
        if gen.device is not None:
            print(f"Device selected: {gen.device}")
            if gen.device.type == "cuda":
                print(f"CUDA available: {torch.cuda.is_available()} | Name: {torch.cuda.get_device_name(0)}")
        else:
            print("Torch not available or no device.")
    except Exception as e:
        print(f"Torch check error: {e}")

    start = datetime(2024, 1, 2)
    # Generate a small number of bars to exercise sampling
    bars = []
    for i in range(10):
        ts = start.replace(hour=0, minute=i)
        bars.append(gen.generate_bar(ts))
    print(f"Generated {len(bars)} bars. Sample close: {bars[-1]['close']}")


if __name__ == "__main__":
    main()
