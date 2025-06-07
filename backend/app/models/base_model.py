"""
Base model class for LifeLens ML models
Defines the common interface and functionality for all models
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import time
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.logging import LoggerMixin


class BaseModel(ABC, LoggerMixin):
    """
    Abstract base class for all LifeLens ML models.
    Provides common functionality and enforces interface compliance.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_id = str(uuid.uuid4())
        self.version = "1.0.0"
        self.created_at = datetime.utcnow()
        self.last_updated = None
        self.training_metadata = {}
        self.is_trained = False
        self._model = None
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Any, **kwargs) -> Any:
        """Make predictions. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state from serialization. Must be implemented by subclasses."""
        pass
    
    def get_version(self) -> str:
        """Get model version."""
        return self.version
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "is_trained": self.is_trained,
            "training_metadata": self.training_metadata,
            "model_type": self.__class__.__name__
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the model."""
        try:
            # Basic health check - ensure model is loaded and responsive
            start_time = time.time()
            
            # Create dummy input for testing
            test_input = self._create_test_input()
            if test_input is not None:
                # Try a prediction to ensure model is working
                _ = await self.predict(test_input)
            
            response_time = (time.time() - start_time) * 1000  # in ms
            
            return {
                "status": "healthy",
                "model_name": self.model_name,
                "version": self.version,
                "is_trained": self.is_trained,
                "response_time_ms": round(response_time, 2),
                "last_check": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Health check failed for {self.model_name}: {str(e)}")
            return {
                "status": "unhealthy",
                "model_name": self.model_name,
                "version": self.version,
                "error": str(e),
                "last_check": datetime.utcnow().isoformat()
            }
    
    def _create_test_input(self) -> Optional[Any]:
        """Create test input for health checks. Override in subclasses."""
        return None
    
    def update_training_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update training metadata."""
        self.training_metadata.update(metadata)
        self.last_updated = datetime.utcnow()
    
    def mark_as_trained(self) -> None:
        """Mark model as trained."""
        self.is_trained = True
        self.last_updated = datetime.utcnow()
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data format. Override in subclasses for specific validation."""
        if input_data is None:
            return False
        return True
    
    def preprocess_input(self, input_data: Any) -> Any:
        """Preprocess input data. Override in subclasses for specific preprocessing."""
        return input_data
    
    def postprocess_output(self, output_data: Any) -> Any:
        """Postprocess output data. Override in subclasses for specific postprocessing."""
        return output_data


class MLModelMixin:
    """Mixin for ML-specific functionality."""
    
    def __init__(self):
        self.feature_names: List[str] = []
        self.feature_importance: Optional[np.ndarray] = None
        self.training_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {}
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        if self.feature_importance is None or not self.feature_names:
            return None
        
        return dict(zip(self.feature_names, self.feature_importance))
    
    def add_training_epoch(self, epoch_data: Dict[str, Any]) -> None:
        """Add training epoch data to history."""
        epoch_data["timestamp"] = datetime.utcnow().isoformat()
        self.training_history.append(epoch_data)
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update model performance metrics."""
        self.performance_metrics.update(metrics)
        self.last_updated = datetime.utcnow()


class EnsembleModel(BaseModel):
    """Base class for ensemble models."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.models: List[BaseModel] = []
        self.weights: Optional[np.ndarray] = None
        self.ensemble_method = "weighted_average"
    
    def add_model(self, model: BaseModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models.append(model)
        if self.weights is None:
            self.weights = np.array([weight])
        else:
            self.weights = np.append(self.weights, weight)
    
    def normalize_weights(self) -> None:
        """Normalize ensemble weights to sum to 1."""
        if self.weights is not None:
            self.weights = self.weights / np.sum(self.weights)
    
    async def predict_ensemble(self, input_data: Any, **kwargs) -> Any:
        """Make ensemble predictions."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        for model in self.models:
            pred = await model.predict(input_data, **kwargs)
            predictions.append(pred)
        
        return self._combine_predictions(predictions)
    
    def _combine_predictions(self, predictions: List[Any]) -> Any:
        """Combine predictions from multiple models. Override in subclasses."""
        if self.ensemble_method == "weighted_average":
            return self._weighted_average(predictions)
        elif self.ensemble_method == "majority_vote":
            return self._majority_vote(predictions)
        else:
            return self._simple_average(predictions)
    
    def _weighted_average(self, predictions: List[Any]) -> Any:
        """Compute weighted average of predictions."""
        if self.weights is None:
            return self._simple_average(predictions)
        
        weighted_sum = sum(pred * weight for pred, weight in zip(predictions, self.weights))
        return weighted_sum
    
    def _simple_average(self, predictions: List[Any]) -> Any:
        """Compute simple average of predictions."""
        return sum(predictions) / len(predictions)
    
    def _majority_vote(self, predictions: List[Any]) -> Any:
        """Compute majority vote of predictions."""
        # This would be implemented for classification tasks
        from collections import Counter
        votes = Counter(predictions)
        return votes.most_common(1)[0][0]
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for ensemble model."""
        base_health = await super().health_check()
        
        # Check health of component models
        model_health = []
        for i, model in enumerate(self.models):
            health = await model.health_check()
            model_health.append({
                "model_index": i,
                "model_name": model.model_name,
                "status": health["status"]
            })
        
        base_health["ensemble_size"] = len(self.models)
        base_health["component_models"] = model_health
        
        return base_health
    
    def get_state(self) -> Dict[str, Any]:
        """Get ensemble state."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "training_metadata": self.training_metadata,
            "is_trained": self.is_trained,
            "ensemble_method": self.ensemble_method,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "models": [model.get_state() for model in self.models]
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load ensemble state."""
        self.model_name = state["model_name"]
        self.model_id = state["model_id"]
        self.version = state["version"]
        self.created_at = datetime.fromisoformat(state["created_at"])
        if state["last_updated"]:
            self.last_updated = datetime.fromisoformat(state["last_updated"])
        self.training_metadata = state["training_metadata"]
        self.is_trained = state["is_trained"]
        self.ensemble_method = state["ensemble_method"]
        self.weights = np.array(state["weights"]) if state["weights"] else None
        
        # Note: Loading component models would require additional logic
        # to instantiate the correct model types
    
    async def initialize(self) -> None:
        """Initialize ensemble model."""
        self.logger.info(f"Initializing ensemble model: {self.model_name}")
        
        # Initialize component models
        for model in self.models:
            await model.initialize()
        
        self.mark_as_trained()


class TimeSeriesModelMixin:
    """Mixin for time series specific functionality."""
    
    def __init__(self):
        self.sequence_length = 30
        self.forecast_horizon = 7
        self.sampling_frequency = "1H"
        self.seasonal_periods = []
    
    def prepare_sequences(self, data: np.ndarray, sequence_length: Optional[int] = None) -> np.ndarray:
        """Prepare sequences for time series modeling."""
        seq_len = sequence_length or self.sequence_length
        
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i + seq_len])
        
        return np.array(sequences)
    
    def add_time_features(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Add time-based features."""
        features = pd.DataFrame(index=timestamps)
        features["hour"] = timestamps.hour
        features["day_of_week"] = timestamps.dayofweek
        features["day_of_year"] = timestamps.dayofyear
        features["month"] = timestamps.month
        features["quarter"] = timestamps.quarter
        features["is_weekend"] = timestamps.dayofweek.isin([5, 6]).astype(int)
        
        return features 