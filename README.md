# FracFire: Synthetic-to-Real MES Price Generator

FracFire is a research platform for generating high-fidelity synthetic futures data (specifically MES/ES) to train machine learning models. It uses a "Physics Engine" approach where price action is generated tick-by-tick based on market states, sessions, and fractal patterns.

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/fracfire.git
cd fracfire

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 🎮 Developer Playground

Run these scripts to see the system in action:

1.  **Basic Generation Demo**
    ```bash
    python scripts/demo_price_generation.py
    ```
    *Generates a full day with auto-transitions and saves charts to `out/charts/`.*

2.  **Enhanced Features Demo**
    ```bash
    python scripts/demo_enhanced_features.py
    ```
    *Shows segment-based control (e.g., 15-min blocks) and detailed statistical analysis.*

3.  **Custom States Demo**
    ```bash
    python scripts/demo_custom_states.py
    ```
    *Visualizes extreme market states like "Flash Crash", "Melt Up", and "News Spike".*

4.  **Generate Archetypes**
    ```bash
    python scripts/generate_archetypes.py
    ```
    *Creates a library of 1000+ labeled synthetic days (Rally, Range, Breakout, etc.) for ML training.*

## 🏗️ Architecture

The system is built in layers:

1.  **Physics Engine (`lab/generators/`)**:
    *   `PriceGenerator`: Tick-based simulation (0.25 tick size).
    *   `FractalStateManager`: Day/Hour/Minute hierarchical states.
    *   `ChartVisualizer`: Matplotlib-based rendering.

2.  **ML Pipeline (`src/`)**:
    *   `features/`: Feature extraction (rolling windows, etc.).
    *   `behavior/`: Markov/Regime learning.
    *   `models/`: Neural tilt models (PyTorch).
    *   `policy/`: Orchestration and consistency.

## 📚 Documentation

*   [Architecture Overview](docs/ARCHITECTURE.md)
*   [Generator Guide](docs/GENERATOR_GUIDE.md)
*   [Project Management](docs/PROJECT_MANAGEMENT.md)

## 🤝 Contributing

See `docs/PROJECT_MANAGEMENT.md` for the current roadmap and tasks.
