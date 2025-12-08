# FracFire Quick Start Guide

## Prerequisites

- Python 3.10+ installed
- Node.js 18+ and npm 9+ installed
- Git (for cloning)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/wyrmspire/fracfire.git
cd fracfire
```

### Step 2: Set Up Python Environment

```bash
# Create and activate virtual environment
python -m venv .venv

# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
pip install -r api/requirements.txt
```

### Step 3: Install Node.js Dependencies

```bash
npm install
```

### Step 4: Configure Environment

```bash
# Copy environment templates
cp frontend/.env.example frontend/.env.local
cp api/.env.example api/.env

# Edit frontend/.env.local and add your Gemini API key (optional for AI assistant)
# GEMINI_API_KEY=your_key_here
```

### Step 5: Create Data Directories

```bash
mkdir -p data/generated data/uploads models/trained
```

## Running the Application

### Option 1: Run Everything Together (Recommended)

```bash
npm run dev
```

This will start both the backend API server (port 8000) and frontend dev server (port 3000).

### Option 2: Run Separately

**Terminal 1 - Backend:**
```bash
cd api
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Access the Application

- **Frontend GUI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## First Steps

1. **Generate Data**:
   - Go to the "Data" tab
   - Click "Generate Random (1000 bars)"
   - Wait for the data to be generated

2. **View Chart**:
   - Go to the "Chart" tab
   - You should see the generated data visualized

3. **Place Manual Trades**:
   - In Chart tab, click "Long" or "Short" buttons
   - Click on the chart to place a trade
   - Drag the TP/SL levels to adjust

4. **Try the AI Assistant**:
   - Add your Gemini API key to `frontend/.env.local`
   - Use the chat panel on the right
   - Ask questions like "Generate 2000 bars of data for MESM5"

## Troubleshooting

### Backend Won't Start

- Check if port 8000 is already in use
- Verify Python dependencies are installed: `pip list`
- Check for errors in the terminal output

### Frontend Won't Start

- Check if port 3000 is already in use
- Verify Node dependencies: `ls node_modules/`
- Try clearing cache: `rm -rf node_modules package-lock.json && npm install`

### Data Generation Fails

- Check that the Python generator is working: `python scripts/demo_price_generation.py`
- Verify that `src/core/generator/` exists and has all files

### API Returns Errors

- Check backend terminal for Python exceptions
- Verify CORS settings in `api/main.py`
- Check browser console for network errors

## Development Tips

### Hot Reload

Both backend and frontend support hot reload:
- Backend: Edit `api/main.py` and it will auto-reload
- Frontend: Edit `frontend/index.tsx` and browser will auto-refresh

### Adding New Features

See `agents.md` for detailed refactoring ideas and architecture notes.

### Database

Currently uses JSON files for data storage. For production, consider:
- PostgreSQL for structured data
- MongoDB for flexible schemas
- Redis for caching

## Next Steps

- Read `agents.md` for future improvement ideas
- Explore `scripts/` for more generator examples
- Check `docs/` for in-depth documentation
- Review `src/core/` for generator implementation details

## Getting Help

- Check existing GitHub issues
- Review code comments in `api/main.py` and `frontend/index.tsx`
- See FracFire documentation in `docs/`

Happy coding! 🚀
