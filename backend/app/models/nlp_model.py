"""
NLP Model for LifeLens
Simple clinical text analysis
"""
import numpy as np
from typing import Dict, Any
from datetime import datetime
import re

from app.models.base_model import BaseModel
from app.core.config import settings


class NLPModel(BaseModel):
    """Simple NLP model for clinical text analysis."""
    
    def __init__(self):
        super().__init__("nlp_model")
        self.is_trained = False
        
        # Simple medical keyword patterns
        self.medical_keywords = {
            'symptoms': ['pain', 'fever', 'cough', 'fatigue', 'nausea'],
            'conditions': ['diabetes', 'hypertension', 'heart disease', 'stroke'],
            'medications': ['aspirin', 'insulin', 'metformin', 'lisinopril']
        }
    
    async def initialize(self) -> None:
        """Initialize the NLP model."""
        self.logger.info("Initializing NLP model")
        self.mark_as_trained()
        self.logger.info("NLP model initialized")
    
    async def predict(self, input_data: Any, **kwargs) -> Any:
        """Make predictions on input data."""
        if isinstance(input_data, str):
            return await self.analyze(input_data, **kwargs)
        elif isinstance(input_data, dict) and 'text' in input_data:
            analysis_type = input_data.get('analysis_type', 'sentiment')
            return await self.analyze(input_data['text'], analysis_type)
        else:
            raise ValueError("Input data must be a string or dict with 'text' field")
    
    async def analyze(self, text: str, analysis_type: str = "sentiment") -> Dict[str, Any]:
        """Analyze clinical text."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Simple sentiment analysis
        positive_words = ['good', 'normal', 'stable', 'improving', 'better']
        negative_words = ['severe', 'critical', 'worse', 'pain', 'emergency']
        
        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        
        if positive_score > negative_score:
            sentiment = "positive"
            confidence = 0.7
        elif negative_score > positive_score:
            sentiment = "negative" 
            confidence = 0.7
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        # Extract medical concepts
        concepts = {}
        for category, keywords in self.medical_keywords.items():
            found = [word for word in keywords if word in text_lower]
            concepts[category] = found
        
        return {
            "sentiment": {
                "classification": sentiment,
                "confidence": confidence,
                "positive_score": positive_score,
                "negative_score": negative_score
            },
            "medical_concepts": concepts,
            "word_count": len(text.split()),
            "analysis_type": analysis_type
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get model state."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "is_trained": self.is_trained
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state."""
        self.model_name = state["model_name"]
        self.model_id = state["model_id"]
        self.version = state["version"]
        self.created_at = datetime.fromisoformat(state["created_at"])
        self.is_trained = state["is_trained"] 