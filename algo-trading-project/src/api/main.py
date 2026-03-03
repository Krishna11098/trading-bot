"""
FastAPI application for algo-trading ML project.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from modeling.model_utils import load_model, predict, get_trading_signal
from utils.logger import get_logger
from utils.config import get_config

logger = get_logger(__name__)
config = get_config()

app = FastAPI(
    title="Algo Trading ML API",
    description="API for ML-based algorithmic trading predictions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    ticker: str
    features: dict


class PredictionResponse(BaseModel):
    ticker: str
    prediction: int
    probability: Optional[List[float]]
    signal: int
    signal_name: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Algo Trading ML API",
        "version": "1.0.0",
        "status": "active"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/config")
async def get_api_config():
    """Get API configuration."""
    return config


@app.get("/models")
async def list_models():
    """List available models."""
    models_dir = Path(config['paths']['models'])
    model_files = list(models_dir.glob("*_model.pkl"))
    tickers = [f.stem.replace("_model", "") for f in model_files]
    
    return {
        "available_models": tickers,
        "count": len(tickers)
    }


@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """
    Make trading prediction for a ticker.
    
    Args:
        request: Prediction request with ticker and features
    
    Returns:
        Prediction response with signal
    """
    try:
        # Load model
        model, scaler, features = load_model(request.ticker)
        
        # Prepare data
        data_dict = {feat: [request.features.get(feat, 0)] for feat in features}
        data = pd.DataFrame(data_dict)
        
        # Make prediction
        predictions, probabilities = predict(model, scaler, features, data)
        
        # Get trading signal
        signal = get_trading_signal(probabilities)
        
        signal_names = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        
        return PredictionResponse(
            ticker=request.ticker,
            prediction=int(predictions[0]),
            probability=probabilities[0].tolist() if probabilities is not None else None,
            signal=signal,
            signal_name=signal_names.get(signal, "UNKNOWN")
        )
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found for ticker: {request.ticker}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tickers")
async def get_default_tickers():
    """Get default ticker list."""
    return {
        "tickers": config['trading']['tickers'],
        "count": len(config['trading']['tickers'])
    }


@app.get("/indicators/{ticker}")
async def get_latest_indicators(ticker: str):
    """
    Get latest technical indicators for a ticker.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Latest indicator values
    """
    try:
        indicators_file = Path(config['paths']['indicators']) / f"{ticker}_features.csv"
        
        if not indicators_file.exists():
            raise HTTPException(status_code=404, detail=f"Indicators not found for {ticker}")
        
        data = pd.read_csv(indicators_file, index_col=0, parse_dates=True)
        latest = data.iloc[-1].to_dict()
        
        return {
            "ticker": ticker,
            "date": str(data.index[-1]),
            "indicators": latest
        }
    
    except Exception as e:
        logger.error(f"Error fetching indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

