# FracFire: Synthetic-to-Real MES Price Generator with AI Agent GUI

FracFire is a comprehensive research platform for generating high-fidelity synthetic futures data (specifically MES/ES) to train machine learning models. It combines a "Physics Engine" approach for price generation with a modern React-based GUI for convenient tool access, charting, and model management.

## 🚀 Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/wyrmspire/fracfire.git
cd fracfire

# Install Python dependencies
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Install Node.js dependencies for the GUI
npm install

# Start the application (runs both backend and frontend)
npm run dev
```

The application will be available at:
- **Frontend GUI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Alternative: Run Backend and Frontend Separately

```bash
# Terminal 1: Backend API
cd api
python main.py

# Terminal 2: Frontend GUI
cd frontend
npm run dev
```

## 🎮 Using the GUI

The FracFire GUI provides a user-friendly interface to:

1. **Generate Synthetic Data**: Create price data with customizable parameters
2. **Chart & Visualize**: Interactive multi-timeframe charting with TradingView-style controls
3. **Manual Trade Lab**: Place and edit trades manually on the chart
4. **Setup Detection**: Run automated setup analysis on your data
5. **Model Training**: Train CNN and other ML models on your datasets
6. **AI Assistant**: Chat with Gemini AI to execute complex workflows

### Key Features

- **Interactive Charting**: Zoom, pan, and switch timeframes (1m, 5m, 15m, 1h)
- **Trade Placement**: Click to place LONG/SHORT trades with draggable TP/SL
- **Data Management**: Upload, generate, and organize datasets
- **Model Training**: Train models and run inference on new data
- **Script Execution**: Run FracFire generator and analysis scripts from the GUI
- **Export/Import**: Save and load trade setups as JSON

## 🎮 Developer Playground (CLI Scripts)

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
