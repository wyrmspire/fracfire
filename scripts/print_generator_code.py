import os
from pathlib import Path

FILES = [
    # Core Generator
    "src/core/generator/engine.py",
    "src/core/generator/states.py",
    "src/core/generator/sampler.py",
    "src/core/generator/fractal_planner.py",
    "src/core/generator/fractal_planner_torch.py",
    "src/core/generator/custom_states.py",
    "src/core/generator/utils.py",
    "src/core/generator/__init__.py",
    
    # Key Scripts
    "scripts/demo_price_generation.py",
    "scripts/calibrate_generator.py",
    "scripts/generate_training_data.py",
]

def main():
    root = Path(__file__).resolve().parents[1]
    
    print("# Generator Code Dump\n")
    
    for rel_path in FILES:
        abs_path = root / rel_path
        if not abs_path.exists():
            print(f"## File: {rel_path} (NOT FOUND)\n")
            continue
            
        print(f"## File: {rel_path}\n")
        print("```python")
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                print(f.read())
        except Exception as e:
            print(f"# Error reading file: {e}")
        print("```\n")
        print("---\n")

if __name__ == "__main__":
    main()
