"""
Generate Code Dump

Splits the codebase into 10 logical markdown files for easier reading/context loading.
"""

import os
from pathlib import Path
from typing import List, Dict

# Define Groups
GROUPS = {
    "01_generators_core": [
        "lab/generators/price_generator.py",
        "lab/generators/fractal_states.py",
        "lab/generators/__init__.py"
    ],
    "02_generators_utils": [
        "lab/generators/custom_states.py",
        "lab/generators/utils.py"
    ],
    "03_visualizers": [
        "lab/visualizers/chart_viz.py",
        "lab/visualizers/__init__.py",
        "scripts/demo_price_generation.py"
    ],
    "04_archetype_scripts": [
        "scripts/generate_archetypes.py",
        "scripts/validate_archetypes.py",
        "scripts/analyze_drift.py"
    ],
    "05_demo_scripts": [
        "scripts/demo_enhanced_features.py",
        "scripts/demo_custom_states.py",
        "scripts/test_installation.py"
    ],
    "06_ml_source": [
        "src/features/builder.py",
        "src/training/data_loader.py",
        "src/data/loader.py",
        "src/models/tilt.py",
        "src/behavior/learner.py",
        "src/policy/orchestrator.py"
    ],
    "07_ml_scripts": [
        "scripts/train_baseline.py",
        "scripts/train_balanced.py",
        "scripts/evaluate_baseline.py",
        "scripts/apply_to_real.py",
        "scripts/apply_optimized.py",
        "scripts/visualize_real.py",
        "scripts/visualize_optimized.py"
    ],
    "08_docs_core": [
        "docs/ARCHITECTURE.md",
        "docs/PROJECT_MANAGEMENT.md",
        "README.md"
    ],
    "09_docs_guides": [
        "docs/GENERATOR_GUIDE.md",
        "docs/SETUP_COMPLETE.md",
        "requirements.txt"
    ],
    "10_agent_instructions": [
        "docs/agent_instructions/phase_1_tasks.md",
        "docs/agent_instructions/legacy_components_directive.md",
        "docs/agent_instructions/style_rules.md",
        "docs/agent_instructions/bootstrapping_checklist.md",
        "docs/agent_instructions/file_replacement_rules.md"
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
