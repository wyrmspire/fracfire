"""
FracFire API - Backend service for the FracFire trading platform
Exposes data generation, model training, and setup detection tools via REST API
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
from pathlib import Path
import json
import asyncio
from datetime import datetime
import uvicorn

# Add project root to path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

app = FastAPI(
    title="FracFire API",
    description="Backend API for FracFire trading platform",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Data Models
# =============================================================================

class GenerateDataRequest(BaseModel):
    symbol: str
    bars: int = 1000
    volatility: float = 2.0
    seed: Optional[int] = None

class TrainModelRequest(BaseModel):
    dataset_id: str
    model_type: str
    epochs: int = 10
    batch_size: int = 32

class SetupAnalysisRequest(BaseModel):
    strategy_id: str
    dataset_id: str
    risk_reward: float = 2.0

class ExecuteScriptRequest(BaseModel):
    script_name: str
    args: Dict[str, Any] = {}

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "FracFire API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# --- Data Generation Endpoints ---

@app.post("/api/data/generate")
async def generate_data(request: GenerateDataRequest):
    """Generate synthetic price data using FracFire generator"""
    try:
        from src.core.generator import PriceGenerator
        from datetime import datetime, timedelta
        import pandas as pd
        
        # Create generator
        gen = PriceGenerator(
            initial_price=5810.0,
            seed=request.seed if request.seed else None
        )
        
        # Calculate start date based on bars requested
        # Assuming 1m bars, calculate how many days needed
        bars_per_day = 1440  # 24 hours * 60 minutes
        days_needed = max(1, (request.bars + bars_per_day - 1) // bars_per_day)
        
        # Generate data
        all_data = []
        for day_offset in range(days_needed):
            start_date = datetime.now() - timedelta(days=days_needed - day_offset)
            df = gen.generate_day(start_date, auto_transition=True)
            all_data.append(df)
        
        # Combine all days
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Limit to requested number of bars
        combined_df = combined_df.head(request.bars)
        
        # Convert to OHLCV format
        data = []
        for _, row in combined_df.iterrows():
            data.append({
                "time": row['time'].isoformat() if hasattr(row['time'], 'isoformat') else str(row['time']),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row.get('volume', 100)),
                "original_symbol": request.symbol
            })
        
        dataset_id = f"ds-{request.symbol}-{int(datetime.now().timestamp())}"
        
        # Save to file
        output_dir = root / "data" / "generated"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{dataset_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(data, f)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "bars": len(data),
            "data": data
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/datasets")
async def list_datasets():
    """List all available datasets"""
    try:
        datasets = []
        
        # Check generated data
        gen_dir = root / "data" / "generated"
        if gen_dir.exists():
            for file in gen_dir.glob("*.json"):
                datasets.append({
                    "id": file.stem,
                    "name": file.stem,
                    "source": "GENERATED",
                    "path": str(file)
                })
        
        # Check uploaded data
        upload_dir = root / "data" / "uploads"
        if upload_dir.exists():
            for file in upload_dir.glob("*.json"):
                datasets.append({
                    "id": file.stem,
                    "name": file.stem,
                    "source": "UPLOAD",
                    "path": str(file)
                })
        
        return {"datasets": datasets}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data/dataset/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get a specific dataset"""
    try:
        # Try generated
        gen_file = root / "data" / "generated" / f"{dataset_id}.json"
        if gen_file.exists():
            with open(gen_file, 'r') as f:
                data = json.load(f)
            return {"id": dataset_id, "data": data}
        
        # Try uploads
        upload_file = root / "data" / "uploads" / f"{dataset_id}.json"
        if upload_file.exists():
            with open(upload_file, 'r') as f:
                data = json.load(f)
            return {"id": dataset_id, "data": data}
        
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file"""
    try:
        contents = await file.read()
        data = json.loads(contents)
        
        # Validate format
        if not isinstance(data, list) or not data:
            raise ValueError("Invalid dataset format")
        
        # Save file
        upload_dir = root / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_id = f"upload-{datetime.now().timestamp()}"
        output_file = upload_dir / f"{dataset_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(data, f)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "bars": len(data)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Model Training Endpoints ---

@app.post("/api/models/train")
async def train_model(request: TrainModelRequest):
    """Train a machine learning model"""
    try:
        # Mock implementation - will be replaced with real training
        model_id = f"model-{request.model_type}-{datetime.now().timestamp()}"
        
        return {
            "success": True,
            "model_id": model_id,
            "model_type": request.model_type,
            "accuracy": 0.85,
            "status": "training"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List all trained models"""
    try:
        models = []
        model_dir = root / "models" / "trained"
        
        if model_dir.exists():
            for file in model_dir.glob("*.pt"):
                models.append({
                    "id": file.stem,
                    "name": file.stem,
                    "type": "CNN",
                    "status": "ready"
                })
        
        return {"models": models}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Setup Analysis Endpoints ---

@app.post("/api/setups/analyze")
async def analyze_setups(request: SetupAnalysisRequest):
    """Analyze setups on a dataset"""
    try:
        # Mock implementation - will be replaced with real analysis
        setups = []
        
        for i in range(10):
            setups.append({
                "id": f"setup-{i}",
                "strategy_id": request.strategy_id,
                "time": datetime.now().isoformat(),
                "type": "LONG" if i % 2 == 0 else "SHORT",
                "entry_price": 5810.0 + i,
                "stop_loss": 5805.0 + i,
                "take_profit": 5820.0 + i,
                "confidence": 0.75
            })
        
        return {
            "success": True,
            "setups": setups
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Script Execution Endpoints ---

@app.post("/api/scripts/execute")
async def execute_script(request: ExecuteScriptRequest):
    """Execute a FracFire script"""
    try:
        scripts_dir = root / "scripts"
        script_file = scripts_dir / request.script_name
        
        if not script_file.exists():
            raise HTTPException(status_code=404, detail="Script not found")
        
        # Mock execution
        return {
            "success": True,
            "output": f"Executed {request.script_name}",
            "script": request.script_name
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scripts")
async def list_scripts():
    """List available scripts"""
    try:
        scripts = []
        scripts_dir = root / "scripts"
        
        for file in scripts_dir.glob("*.py"):
            scripts.append({
                "name": file.name,
                "path": str(file),
                "category": "generator" if "generate" in file.name else "training" if "train" in file.name else "analysis"
            })
        
        return {"scripts": scripts}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Server Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
