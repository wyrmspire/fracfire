"""
Generate Code Dump

Splits the codebase into 10 logical markdown files for easier reading/context loading.
"""

import os
from pathlib import Path
from typing import List, Dict

# Define Groups
GROUPS = {
    "01_core_generator": [
        "src/core/generator/__init__.py",
        "src/core/generator/engine.py",
        "src/core/generator/states.py",
        "src/core/generator/custom_states.py",
        "src/core/generator/utils.py"
    ],
    "02_core_detector": [
        "src/core/detector/__init__.py",
        "src/core/detector/models.py",
        "src/core/detector/indicators.py",
        "src/core/detector/library.py",
        "src/core/detector/features.py",
        "src/core/detector/engine.py"
    ],
    "03_data_loaders": [
        "src/data/loader.py",
        "src/data/__init__.py"
    ],
    "04_ml_features": [
        "src/ml/features/builder.py",
        "src/ml/features/images.py"
    ],
    "05_ml_training": [
        "src/ml/training/data_loader.py",
        "src/ml/training/labeler.py",
        "src/ml/training/factory.py"
    ],
    "06_ml_models": [
        "src/ml/models/tilt.py",
        "src/ml/models/cnn.py",
        "src/ml/models/generative.py"
    ],
    "07_agent_modules": [
        "src/agent/learner.py",
        "src/agent/orchestrator.py",
        "src/agent/tools.py"
    ],
    "08_runner_scripts": [
        "scripts/run_orb_trade_runner.py",
        "scripts/plot_trade_setups.py",
        "scripts/compare_real_vs_generator.py",
        "scripts/analyze_real_vs_synth.py"
    ],
    "09_demo_scripts": [
        "scripts/demo_enhanced_features.py",
        "scripts/demo_custom_states.py",
        "scripts/demo_price_generation.py",
        "scripts/test_installation.py"
    ],
    "10_docs_core": [
        "docs/ARCHITECTURE.md",
        "docs/GENERATOR_DEFAULTS.md",
        "docs/SETUP_ENGINE.md",
        "README.md"
    ],
    "11_docs_guides": [
        "docs/GENERATOR_GUIDE.md",
        "docs/SETUP_COMPLETE.md",
        "docs/PROJECT_MANAGEMENT.md",
        "requirements.txt"
    ],
    "12_agent_instructions": [
        "docs/agent_instructions/phase_1_tasks.md",
        "docs/agent_instructions/legacy_components_directive.md",
        "docs/agent_instructions/style_rules.md",
        "docs/agent_instructions/bootstrapping_checklist.md",
        "docs/agent_instructions/file_replacement_rules.md",
        "docs/agent_instructions/reverse_engineering_directive.md"
    ]
}

def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root # Output to root as requested
    
    print(f"Generating code dump in {output_dir}...")
    
    for group_name, files in GROUPS.items():
        filename = f"code_dump_{group_name}.md"
        filepath = output_dir / filename
        
        print(f"  Writing {filename}...")
        
        with open(filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(f"# Code Dump: {group_name}\n\n")
            
            for rel_path in files:
                abs_path = root / rel_path
                if not abs_path.exists():
                    print(f"    WARNING: File not found: {rel_path}")
                    continue
                    
                outfile.write(f"## File: {rel_path}\n")
                
                # Determine language
                ext = abs_path.suffix.lower()
                lang = "python" if ext == ".py" else "markdown" if ext == ".md" else "text"
                if rel_path == "requirements.txt": lang = "text"
                
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
