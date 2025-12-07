"""
Generate Generator Code Dump

Splits the generator-related codebase into logical markdown files in the root directory,
following the same pattern as generate_code_dump.py.
"""

import os
from pathlib import Path

# Define Generator-specific Groups
GROUPS = {
    "01_generator_core": [
        "src/core/generator/__init__.py",
        "src/core/generator/engine.py",
        "src/core/generator/states.py",
        "src/core/generator/sampler.py",
        "src/core/generator/fractal_planner.py",
        "src/core/generator/fractal_planner_torch.py",
        "src/core/generator/custom_states.py",
        "src/core/generator/utils.py"
    ],
    "02_generator_scripts": [
        "scripts/demo_price_generation.py",
        "scripts/calibrate_generator.py",
        "scripts/generate_training_data.py",
        "scripts/generate_month_chart.py",
        "scripts/measure_market_physics.py"
    ]
}

def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root # Output to root
    
    print(f"Generating generator code dump in {output_dir}...")
    
    for group_name, files in GROUPS.items():
        filename = f"generator_dump_{group_name}.md"
        filepath = output_dir / filename
        
        print(f"  Writing {filename}...")
        
        with open(filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(f"# Generator Dump: {group_name}\n\n")
            
            for rel_path in files:
                abs_path = root / rel_path
                if not abs_path.exists():
                    print(f"    WARNING: File not found: {rel_path}")
                    continue
                    
                outfile.write(f"## File: {rel_path}\n")
                
                # Determine language
                ext = abs_path.suffix.lower()
                lang = "python" if ext == ".py" else "markdown" if ext == ".md" else "text"
                
                outfile.write(f"```{lang}\n")
                try:
                    with open(abs_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"Error reading file: {e}")
                outfile.write("\n```\n\n")
                outfile.write("---\n\n")
                
    print("Done!")

if __name__ == "__main__":
    main()
