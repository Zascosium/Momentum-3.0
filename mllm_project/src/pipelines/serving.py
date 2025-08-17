"""
Model Serving API

This module provides a REST API for model inference.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from datetime import datetime
import json
import time

# Databricks compatibility setup
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

if 'DATABRICKS_RUNTIME_VERSION' in os.environ:
    current_path = Path(__file__).parent
    while current_path != current_path.parent:
        if (current_path / 'src').exists():
            sys.path.insert(0, str(current_path / 'src'))
            break
        current_path = current_path.parent

# Core dependencies with fallbacks
try:
    import torch
except ImportError:
    torch = None

# FastAPI imports with better error handling
try:
    from fastapi import FastAPI, HTTPException, Depends, Security
    from fastapi.security import APIKeyHeader
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create mock classes for when FastAPI is not available
    class FastAPI:
        def __init__(self, *args, **kwargs): pass
    class HTTPException(Exception): pass
    class BaseModel: pass
    class Field: pass
    def Depends(*args, **kwargs): pass
    def Security(*args, **kwargs): pass
    class APIKeyHeader: 
        def __init__(self, *args, **kwargs): pass
    class CORSMiddleware: pass
    uvicorn = None

# Import project modules with fallbacks
try:
    from models.multimodal_model import MultimodalLLM
except (ImportError, ValueError):
    try:
        from ..models.multimodal_model import MultimodalLLM
    except (ImportError, ValueError):
        try:
            # Direct import for Databricks
            import importlib.util
            model_path = parent_dir / "models" / "multimodal_model.py"
            if model_path.exists():
                spec = importlib.util.spec_from_file_location("multimodal_model", model_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    MultimodalLLM = module.MultimodalLLM
                else:
                    MultimodalLLM = None
            else:
                MultimodalLLM = None
        except Exception:
            MultimodalLLM = None

try:
    from utils.inference_utils import create_inference_pipeline
except ImportError:
    try:
        from ..utils.inference_utils import create_inference_pipeline
    except ImportError:
        def create_inference_pipeline(*args, **kwargs):
            return None

logger = logging.getLogger(__name__)


# Pydantic models for API
if FASTAPI_AVAILABLE:
    
    class TimeSeriesInput(BaseModel):
        """Input model for time series data."""
        data: List[List[float]] = Field(..., description="Time series data as 2D array")
        
    class TextInput(BaseModel):
        """Input model for text prompt."""
        prompt: str = Field(..., description="Text prompt for generation")
        temperature: float = Field(0.8, ge=0.1, le=2.0, description="Generation temperature")
        max_length: int = Field(100, ge=10, le=500, description="Maximum generation length")
        
    class GenerationRequest(BaseModel):
        """Request model for text generation."""
        time_series: Optional[TimeSeriesInput] = Field(None, description="Optional time series input")
        text: TextInput = Field(..., description="Text generation parameters")
        
    class GenerationResponse(BaseModel):
        """Response model for text generation."""
        generated_text: str = Field(..., description="Generated text")
        generation_time: float = Field(..., description="Generation time in seconds")
        timestamp: str = Field(..., description="Generation timestamp")
        
    class BatchGenerationRequest(BaseModel):
        """Request model for batch generation."""
        requests: List[GenerationRequest] = Field(..., description="List of generation requests")
        
    class BatchGenerationResponse(BaseModel):
        """Response model for batch generation."""
        results: List[GenerationResponse] = Field(..., description="List of generation results")
        total_time: float = Field(..., description="Total processing time")
        
    class HealthResponse(BaseModel):
        """Response model for health check."""
        status: str = Field(..., description="Service status")
        model_loaded: bool = Field(..., description="Whether model is loaded")
        device: str = Field(..., description="Computing device")
        timestamp: str = Field(..., description="Current timestamp")


class ModelServingAPI:
    """
    REST API for model serving.
    """
    
    def __init__(self, config: Dict[str, Any], model_path: str,
                 enable_cors: bool = True, api_key: Optional[str] = None):
        """
        Initialize the serving API.
        
        Args:
            config: Model configuration
            model_path: Path to trained model
            enable_cors: Enable CORS support
            api_key: Optional API key for authentication
        """
        # Check critical dependencies
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for model serving. Install with: pip install fastapi uvicorn")
        if torch is None:
            raise ImportError("PyTorch is required for model serving. Install with: pip install torch")
        if MultimodalLLM is None:
            raise ImportError("MultimodalLLM model not available. Check model imports.")
        
        self.config = config
        self.model_path = Path(model_path)
        self.api_key = api_key
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Multimodal LLM API",
            description="REST API for multimodal language model inference",
            version="1.0.0"
        )
        
        # Setup CORS if enabled
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Load model
        self.model = None
        self.inference_engine = None
        self._load_model()
        
        # Setup routes
        self._setup_routes()
        
        # Track statistics
        self.stats = {
            'requests_served': 0,
            'total_generation_time': 0,
            'start_time': datetime.now()
        }
    
    def _load_model(self):
        """Load model for inference."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Try optimized inference engine
            self.inference_engine = create_inference_pipeline(
                model_path=str(self.model_path),
                config_path=None,
                device=self.device,
                optimization_level="fast"
            )
            logger.info("Using optimized inference engine")
            
        except Exception as e:
            logger.warning(f"Failed to load optimized engine: {e}")
            
            # Fallback to direct model loading
            if self.model_path.is_file():
                checkpoint_path = self.model_path
            elif self.model_path.is_dir():
                checkpoint_path = self.model_path / 'best_model.pt'
                if not checkpoint_path.exists():
                    checkpoint_path = self.model_path / 'final_model.pt'
            else:
                raise ValueError(f"Model not found at {self.model_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model_config = checkpoint.get('config', self.config)
            
            self.model = MultimodalLLM(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded directly")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # API key dependency
        api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
        
        async def verify_api_key(api_key: str = Security(api_key_header)):
            if self.api_key and api_key != self.api_key:
                raise HTTPException(status_code=403, detail="Invalid API key")
            return api_key
        
        # Health check endpoint
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Check service health."""
            return HealthResponse(
                status="healthy",
                model_loaded=self.model is not None or self.inference_engine is not None,
                device=str(self.device),
                timestamp=datetime.now().isoformat()
            )
        
        # Single generation endpoint
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate(
            request: GenerationRequest,
            api_key: str = Depends(verify_api_key) if self.api_key else None
        ):
            """Generate text from optional time series and text prompt."""
            try:
                start_time = time.time()
                
                # Prepare inputs
                time_series = None
                if request.time_series:
                    time_series = np.array(request.time_series.data)
                
                # Generate text
                generated_text = self._generate(
                    time_series=time_series,
                    prompt=request.text.prompt,
                    temperature=request.text.temperature,
                    max_length=request.text.max_length
                )
                
                generation_time = time.time() - start_time
                
                # Update statistics
                self.stats['requests_served'] += 1
                self.stats['total_generation_time'] += generation_time
                
                return GenerationResponse(
                    generated_text=generated_text,
                    generation_time=generation_time,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Batch generation endpoint
        @self.app.post("/generate_batch", response_model=BatchGenerationResponse)
        async def generate_batch(
            request: BatchGenerationRequest,
            api_key: str = Depends(verify_api_key) if self.api_key else None
        ):
            """Generate text for multiple requests."""
            try:
                start_time = time.time()
                results = []
                
                for req in request.requests:
                    # Prepare inputs
                    time_series = None
                    if req.time_series:
                        time_series = np.array(req.time_series.data)
                    
                    # Generate text
                    gen_start = time.time()
                    generated_text = self._generate(
                        time_series=time_series,
                        prompt=req.text.prompt,
                        temperature=req.text.temperature,
                        max_length=req.text.max_length
                    )
                    gen_time = time.time() - gen_start
                    
                    results.append(GenerationResponse(
                        generated_text=generated_text,
                        generation_time=gen_time,
                        timestamp=datetime.now().isoformat()
                    ))
                
                total_time = time.time() - start_time
                
                # Update statistics
                self.stats['requests_served'] += len(request.requests)
                self.stats['total_generation_time'] += total_time
                
                return BatchGenerationResponse(
                    results=results,
                    total_time=total_time
                )
                
            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Statistics endpoint
        @self.app.get("/stats")
        async def get_stats(
            api_key: str = Depends(verify_api_key) if self.api_key else None
        ):
            """Get API statistics."""
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()
            avg_time = (
                self.stats['total_generation_time'] / self.stats['requests_served']
                if self.stats['requests_served'] > 0 else 0
            )
            
            return {
                "requests_served": self.stats['requests_served'],
                "total_generation_time": self.stats['total_generation_time'],
                "average_generation_time": avg_time,
                "uptime_seconds": uptime,
                "start_time": self.stats['start_time'].isoformat()
            }
        
        # Model info endpoint
        @self.app.get("/model_info")
        async def get_model_info():
            """Get model information."""
            if self.model:
                model_stats = self.model.get_memory_usage()
            else:
                model_stats = {"status": "Using inference engine"}
            
            return {
                "model_path": str(self.model_path),
                "device": str(self.device),
                "model_stats": model_stats
            }
    
    def _generate(self, time_series: Optional[np.ndarray], prompt: str,
                 temperature: float = 0.8, max_length: int = 100) -> str:
        """Generate text using the model."""
        
        if self.inference_engine:
            # Use inference engine
            result = self.inference_engine.generate_text(
                time_series=time_series,
                text_prompt=prompt,
                temperature=temperature,
                max_length=max_length
            )
            return result.generated_text if hasattr(result, 'generated_text') else str(result)
        
        elif self.model:
            # Use direct model
            with torch.no_grad():
                # Prepare inputs
                if time_series is not None:
                    ts_tensor = torch.tensor(time_series, dtype=torch.float32).unsqueeze(0).to(self.device)
                    ts_mask = torch.ones(ts_tensor.shape[0], ts_tensor.shape[1], dtype=torch.bool).to(self.device)
                else:
                    ts_tensor = None
                    ts_mask = None
                
                # Simple generation (mock for demonstration)
                # In production, properly tokenize prompt and generate
                if hasattr(self.model, 'generate'):
                    generated = self.model.generate(
                        time_series=ts_tensor,
                        ts_attention_mask=ts_mask,
                        text_prompt=prompt,
                        temperature=temperature,
                        max_length=max_length
                    )
                    return generated
                else:
                    # Mock generation
                    return f"{prompt} [Generated text based on the input data]"
        
        else:
            raise RuntimeError("Model not loaded")
    
    def run(self, host: str = "0.0.0.0", port: int = 8080,
           workers: int = 1, reload: bool = False):
        """
        Run the API server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            workers: Number of worker processes
            reload: Enable auto-reload
        """
        logger.info(f"Starting API server on {host}:{port}")
        
        if self.api_key:
            logger.info("API key authentication enabled")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level="info"
        )
