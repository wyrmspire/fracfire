"""
Generate Setup & Indicator Code Dump

Splits the setup/detector-related codebase into logical markdown files in the root directory.
"""

import os
from pathlib import Path

# Define Setup-specific Groups
GROUPS = {
    "01_detector_core": [
        "src/core/detector/__init__.py",
        "src/core/detector/engine.py",
        "src/core/detector/models.py", 
        "src/core/detector/indicators.py",
        "src/core/detector/library.py",
        "src/core/detector/features.py"
    ],
    "02_detector_scripts": [
        "scripts/run_orb_trade_runner.py",
        "scripts/plot_trade_setups.py",
        "scripts/demo_all_setups.py",
        "scripts/benchmark_detection.py",
        "scripts/demo_orb_setup.py"
    ]
}

def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root # Output to root
    
    print(f"Generating setup/detector code dump in {output_dir}...")
    
    for group_name, files in GROUPS.items():
        filename = f"setup_dump_{group_name}.md"
        filepath = output_dir / filename
        
        print(f"  Writing {filename}...")
        
        with open(filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(f"# Setup Dump: {group_name}\n\n")
            
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
