"""
Model Manager for LifeLens ML models
Handles model loading, caching, and lifecycle management
"""
import asyncio
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import LoggerMixin
from app.models.survival_model import SurvivalModel
from app.models.event_prediction_model import EventPredictionModel
from app.models.time_series_model import TimeSeriesModel
from app.models.nlp_model import NLPModel
from app.models.fairness_monitor import FairnessMonitor


class ModelCache:
    """Thread-safe model cache with LRU eviction."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get model from cache."""
        async with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]["model"]
            return None
    
    async def put(self, key: str, model: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Put model in cache with LRU eviction."""
        async with self._lock:
            # Evict least recently used if cache is full
            if len(self.cache) >= self.max_size:
                lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[lru_key]
                del self.access_times[lru_key]
            
            self.cache[key] = {
                "model": model,
                "metadata": metadata or {},
                "cached_at": time.time()
            }
            self.access_times[key] = time.time()
    
    async def remove(self, key: str) -> None:
        """Remove model from cache."""
        async with self._lock:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all models from cache."""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": 0.0,  # Would need request tracking for real hit rate
            "models": list(self.cache.keys())
        }


class ModelManager(LoggerMixin):
    """
    Central model manager for LifeLens system.
    Handles model loading, caching, prediction routing, and lifecycle management.
    """
    
    def __init__(self):
        self.cache = ModelCache(max_size=settings.MODEL_CACHE_SIZE)
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.fairness_monitor = None
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the model manager and load core models."""
        async with self._lock:
            if self._initialized:
                return
            
            self.logger.info("Initializing model manager")
            
            # Create models directory if it doesn't exist
            Path(settings.MODEL_PATH).mkdir(parents=True, exist_ok=True)
            
            # Initialize fairness monitor
            self.fairness_monitor = FairnessMonitor()
            
            # Load core models
            await self._load_core_models()
            
            self._initialized = True
            self.logger.info("Model manager initialized successfully")
    
    async def _load_core_models(self) -> None:
        """Load core models required for the system."""
        core_models = {
            "survival_model": SurvivalModel(),
            "event_prediction_model": EventPredictionModel(),
            "time_series_model": TimeSeriesModel(),
            "nlp_model": NLPModel()
        }
        
        for model_name, model_instance in core_models.items():
            try:
                await self._load_model(model_name, model_instance)
                self.logger.info(f"Loaded core model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load core model {model_name}: {str(e)}")
                # For demo purposes, we'll create synthetic models if real ones don't exist
                await self._create_synthetic_model(model_name, model_instance)
    
    async def _load_model(self, model_name: str, model_instance: Any) -> None:
        """Load a specific model from disk or initialize it."""
        model_path = settings.get_model_path(model_name)
        
        if model_path.exists():
            # Load existing model
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                model_instance.load_state(model_data)
        else:
            # Initialize new model (this would normally involve training)
            await model_instance.initialize()
        
        # Cache the model
        await self.cache.put(model_name, model_instance, {
            "loaded_at": time.time(),
            "model_type": type(model_instance).__name__
        })
        
        self.models[model_name] = model_instance
    
    async def _create_synthetic_model(self, model_name: str, model_instance: Any) -> None:
        """Create a synthetic model for demonstration purposes."""
        self.logger.info(f"Creating synthetic model for {model_name}")
        
        # Initialize with synthetic data
        await model_instance.initialize()
        
        # Save the model
        await self.save_model(model_name, model_instance)
        
        # Cache the model
        await self.cache.put(model_name, model_instance, {
            "loaded_at": time.time(),
            "model_type": type(model_instance).__name__,
            "synthetic": True
        })
        
        self.models[model_name] = model_instance
    
    async def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model from cache or load it."""
        # Try cache first
        model = await self.cache.get(model_name)
        if model:
            return model
        
        # Try to load from disk
        if model_name in self.models:
            return self.models[model_name]
        
        self.logger.warning(f"Model not found: {model_name}")
        return None
    
    async def predict_survival(
        self,
        patient_data: Dict[str, Any],
        horizons: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Predict survival probabilities."""
        model = await self.get_model("survival_model")
        if not model:
            raise RuntimeError("Survival model not available")
        
        horizons = horizons or settings.SURVIVAL_HORIZONS
        
        # Prepare input data
        input_data = self._prepare_survival_input(patient_data)
        
        # Make prediction
        start_time = time.time()
        prediction = await model.predict(input_data, horizons)
        prediction_time = time.time() - start_time
        
        # Monitor prediction for fairness
        if self.fairness_monitor:
            await self.fairness_monitor.check_survival_prediction(
                patient_data, prediction
            )
        
        # Log prediction
        self.logger.info(
            "Survival prediction completed",
            patient_id=patient_data.get("patient_id", "unknown"),
            horizons=horizons,
            processing_time=prediction_time
        )
        
        return {
            "survival_probabilities": prediction,
            "horizons": horizons,
            "confidence_interval": settings.CONFIDENCE_INTERVAL,
            "model_version": model.get_version(),
            "processing_time": prediction_time
        }
    
    async def predict_events(
        self,
        patient_data: Dict[str, Any],
        event_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Predict health event probabilities."""
        model = await self.get_model("event_prediction_model")
        if not model:
            raise RuntimeError("Event prediction model not available")
        
        event_types = event_types or settings.EVENT_TYPES
        
        # Prepare input data
        input_data = self._prepare_event_input(patient_data)
        
        # Make prediction
        start_time = time.time()
        predictions = await model.predict(input_data, event_types)
        prediction_time = time.time() - start_time
        
        # Monitor prediction for fairness
        if self.fairness_monitor:
            await self.fairness_monitor.check_event_prediction(
                patient_data, predictions
            )
        
        # Log prediction
        self.logger.info(
            "Event prediction completed",
            patient_id=patient_data.get("patient_id", "unknown"),
            event_types=event_types,
            processing_time=prediction_time
        )
        
        return {
            "event_probabilities": predictions,
            "event_types": event_types,
            "model_version": model.get_version(),
            "processing_time": prediction_time
        }
    
    async def predict_time_series(
        self,
        time_series_data: Dict[str, Any],
        forecast_horizon: int = 30
    ) -> Dict[str, Any]:
        """Predict future values for time series data."""
        model = await self.get_model("time_series_model")
        if not model:
            raise RuntimeError("Time series model not available")
        
        # Prepare input data
        input_data = self._prepare_time_series_input(time_series_data)
        
        # Make prediction
        start_time = time.time()
        forecast = await model.predict(input_data, forecast_horizon)
        prediction_time = time.time() - start_time
        
        self.logger.info(
            "Time series prediction completed",
            subject_id=time_series_data.get("subject_id", "unknown"),
            forecast_horizon=forecast_horizon,
            processing_time=prediction_time
        )
        
        return {
            "forecast": forecast,
            "forecast_horizon": forecast_horizon,
            "model_version": model.get_version(),
            "processing_time": prediction_time
        }
    
    async def analyze_text(
        self,
        clinical_text: str,
        analysis_type: str = "sentiment"
    ) -> Dict[str, Any]:
        """Analyze clinical text using NLP models."""
        model = await self.get_model("nlp_model")
        if not model:
            raise RuntimeError("NLP model not available")
        
        # Make prediction
        start_time = time.time()
        analysis = await model.analyze(clinical_text, analysis_type)
        processing_time = time.time() - start_time
        
        self.logger.info(
            "Text analysis completed",
            text_length=len(clinical_text),
            analysis_type=analysis_type,
            processing_time=processing_time
        )
        
        return {
            "analysis": analysis,
            "analysis_type": analysis_type,
            "model_version": model.get_version(),
            "processing_time": processing_time
        }
    
    def _prepare_survival_input(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Prepare patient data for survival model input."""
        # This would implement proper feature engineering
        # For now, create a synthetic feature vector
        features = [
            patient_data.get("age", 50),
            1 if patient_data.get("gender") == "male" else 0,
            patient_data.get("bmi", 25),
            patient_data.get("systolic_bp", 120),
            patient_data.get("cholesterol", 200),
            patient_data.get("smoking", 0),
            patient_data.get("diabetes", 0),
            patient_data.get("heart_disease", 0)
        ]
        return np.array(features).reshape(1, -1)
    
    def _prepare_event_input(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Prepare patient data for event prediction input."""
        # Similar to survival input but might have different features
        return self._prepare_survival_input(patient_data)
    
    def _prepare_time_series_input(self, time_series_data: Dict[str, Any]) -> np.ndarray:
        """Prepare time series data for model input."""
        # Extract time series values
        series = time_series_data.get("values", [])
        if not series:
            # Generate synthetic time series for demo
            series = np.random.normal(100, 10, 100).tolist()
        
        return np.array(series)
    
    async def save_model(self, model_name: str, model: Any) -> None:
        """Save a model to disk."""
        model_path = settings.get_model_path(model_name)
        
        try:
            model_data = model.get_state() if hasattr(model, 'get_state') else model
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to save model {model_name}: {str(e)}")
            raise
    
    async def reload_model(self, model_name: str) -> None:
        """Reload a model from disk."""
        await self.cache.remove(model_name)
        
        if model_name in self.models:
            model_instance = self.models[model_name]
            await self._load_model(model_name, model_instance)
            self.logger.info(f"Model reloaded: {model_name}")
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model."""
        model = await self.get_model(model_name)
        if not model:
            return None
        
        return {
            "name": model_name,
            "type": type(model).__name__,
            "version": getattr(model, "get_version", lambda: "unknown")(),
            "loaded": True,
            "metadata": self.model_metadata.get(model_name, {})
        }
    
    async def get_all_models_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded models."""
        models_info = []
        for model_name in self.models.keys():
            info = await self.get_model_info(model_name)
            if info:
                models_info.append(info)
        return models_info
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up model manager")
        await self.cache.clear()
        self.models.clear()
        self.model_metadata.clear()
        self._initialized = False
    
    def is_initialized(self) -> bool:
        """Check if model manager is initialized."""
        return self._initialized
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all models."""
        health_status = {
            "overall_status": "healthy",
            "models": {},
            "cache_stats": self.cache.get_stats(),
            "initialized": self._initialized
        }
        
        for model_name in self.models.keys():
            try:
                model = await self.get_model(model_name)
                if model and hasattr(model, "health_check"):
                    model_health = await model.health_check()
                else:
                    model_health = {"status": "loaded", "available": model is not None}
                
                health_status["models"][model_name] = model_health
            except Exception as e:
                health_status["models"][model_name] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
        
        return health_status 