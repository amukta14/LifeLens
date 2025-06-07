"""
API routes for LifeLens: Predictive Health and Survival Insight System
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
import logging

from ..core.model_manager import ModelManager
from ..models.fairness_monitor import FairnessMonitor

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize model manager and fairness monitor
model_manager = ModelManager()
fairness_monitor = FairnessMonitor()

# Request/Response Models
class HealthRecord(BaseModel):
    """Health record input for predictions"""
    age: float = Field(..., description="Age in years")
    gender: str = Field(..., description="Gender (M/F/Other)")
    race: Optional[str] = Field(None, description="Race/ethnicity")
    height_cm: float = Field(..., description="Height in centimeters")
    weight_kg: float = Field(..., description="Weight in kilograms")
    blood_pressure_systolic: float = Field(..., description="Systolic blood pressure")
    blood_pressure_diastolic: float = Field(..., description="Diastolic blood pressure")
    cholesterol: float = Field(..., description="Total cholesterol mg/dL")
    glucose: float = Field(..., description="Blood glucose mg/dL")
    heart_rate: float = Field(..., description="Resting heart rate")
    smoking: bool = Field(..., description="Current smoking status")
    diabetes: bool = Field(..., description="Diabetes history")
    hypertension: bool = Field(..., description="Hypertension history")
    family_history_heart: bool = Field(..., description="Family history of heart disease")
    exercise_hours_week: float = Field(..., description="Exercise hours per week")
    alcohol_drinks_week: float = Field(..., description="Alcohol drinks per week")

class ClinicalNote(BaseModel):
    """Clinical note for NLP analysis"""
    text: str = Field(..., description="Clinical note text")
    note_type: str = Field(..., description="Type of clinical note")
    date: datetime = Field(..., description="Note date")

class TimeSeriesData(BaseModel):
    """Time series health data"""
    timestamps: List[datetime] = Field(..., description="Timestamps")
    values: List[float] = Field(..., description="Health metric values")
    metric_type: str = Field(..., description="Type of health metric")

class PredictionRequest(BaseModel):
    """Complete prediction request"""
    health_record: HealthRecord
    clinical_notes: Optional[List[ClinicalNote]] = None
    time_series: Optional[Dict[str, TimeSeriesData]] = None
    prediction_horizon_years: Optional[int] = Field(5, description="Prediction horizon in years")

class SurvivalPrediction(BaseModel):
    """Survival prediction response"""
    survival_probability_5yr: float
    survival_probability_10yr: float
    survival_curve: Dict[str, List[float]]
    confidence_intervals: Dict[str, Dict[str, List[float]]]
    risk_factors: List[Dict[str, Any]]

class EventPrediction(BaseModel):
    """Health event prediction response"""
    cardiac_arrest_risk: float
    stroke_risk: float
    diabetes_onset_risk: float
    hypertension_risk: float
    feature_importance: Dict[str, float]
    risk_timeline: Dict[str, List[float]]

class FairnessReport(BaseModel):
    """Fairness monitoring report"""
    overall_fairness_score: float
    bias_alerts: List[Dict[str, Any]]
    demographic_parity: Dict[str, float]
    recommendations: List[str]

# API Endpoints

@router.post("/predict/survival", response_model=SurvivalPrediction)
async def predict_survival(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict long-term survival probabilities
    """
    try:
        logger.info(f"Survival prediction request for age {request.health_record.age}")
        
        # Get survival model
        survival_model = model_manager.get_model("survival")
        
        # Prepare features
        features = _prepare_features(request.health_record)
        
        # Get prediction
        prediction = survival_model.predict_survival(features, request.prediction_horizon_years)
        
        # Log prediction for fairness monitoring
        background_tasks.add_task(
            _log_prediction_for_fairness,
            "survival",
            features,
            prediction,
            request.health_record
        )
        
        return SurvivalPrediction(**prediction)
        
    except Exception as e:
        logger.error(f"Survival prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/events", response_model=EventPrediction)
async def predict_health_events(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict health event probabilities
    """
    try:
        logger.info(f"Health event prediction request for age {request.health_record.age}")
        
        # Get event prediction model
        event_model = model_manager.get_model("event_prediction")
        
        # Prepare features
        features = _prepare_features(request.health_record)
        
        # Get prediction
        prediction = event_model.predict_events(features)
        
        # Log prediction for fairness monitoring
        background_tasks.add_task(
            _log_prediction_for_fairness,
            "events",
            features,
            prediction,
            request.health_record
        )
        
        return EventPrediction(**prediction)
        
    except Exception as e:
        logger.error(f"Event prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/analyze/clinical-notes")
async def analyze_clinical_notes(notes: List[ClinicalNote]):
    """
    Analyze clinical notes using NLP
    """
    try:
        logger.info(f"Clinical note analysis for {len(notes)} notes")
        
        # Get NLP model
        nlp_model = model_manager.get_model("nlp")
        
        results = []
        for note in notes:
            analysis = nlp_model.analyze_text(note.text)
            results.append({
                "note_type": note.note_type,
                "date": note.date,
                "sentiment": analysis.get("sentiment", {}),
                "medical_concepts": analysis.get("medical_concepts", []),
                "risk_indicators": analysis.get("risk_indicators", [])
            })
        
        return {"analyses": results}
        
    except Exception as e:
        logger.error(f"Clinical note analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/predict/time-series")
async def predict_time_series(data: Dict[str, TimeSeriesData]):
    """
    Predict future health metrics from time series data
    """
    try:
        logger.info(f"Time series prediction for {len(data)} metrics")
        
        # Get time series model
        ts_model = model_manager.get_model("time_series")
        
        predictions = {}
        for metric_name, ts_data in data.items():
            prediction = ts_model.forecast(
                values=ts_data.values,
                timestamps=ts_data.timestamps,
                metric_type=ts_data.metric_type
            )
            predictions[metric_name] = prediction
        
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Time series prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/fairness/report", response_model=FairnessReport)
async def get_fairness_report():
    """
    Get current fairness monitoring report
    """
    try:
        logger.info("Generating fairness report")
        
        report = fairness_monitor.generate_report()
        
        return FairnessReport(
            overall_fairness_score=report["overall_fairness_score"],
            bias_alerts=report["bias_alerts"],
            demographic_parity=report["demographic_parity"],
            recommendations=report["recommendations"]
        )
        
    except Exception as e:
        logger.error(f"Fairness report error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/models/status")
async def get_model_status():
    """
    Get status of all models
    """
    try:
        status = model_manager.get_models_status()
        return {"models": status}
        
    except Exception as e:
        logger.error(f"Model status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/models/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """
    Trigger model retraining
    """
    try:
        logger.info("Triggering model retraining")
        
        # Schedule retraining in background
        background_tasks.add_task(model_manager.retrain_all_models)
        
        return {"message": "Model retraining initiated"}
        
    except Exception as e:
        logger.error(f"Model retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

# Helper functions

def _prepare_features(health_record: HealthRecord) -> np.ndarray:
    """
    Convert health record to feature vector
    """
    # Calculate BMI
    bmi = health_record.weight_kg / (health_record.height_cm / 100) ** 2
    
    # Create feature vector (matches training data format)
    features = np.array([
        health_record.age,
        1 if health_record.gender == 'M' else 0,  # gender_male
        bmi,
        health_record.blood_pressure_systolic,
        health_record.blood_pressure_diastolic,
        health_record.cholesterol,
        health_record.glucose,
        health_record.heart_rate,
        1 if health_record.smoking else 0,
        1 if health_record.diabetes else 0,
        1 if health_record.hypertension else 0,
        1 if health_record.family_history_heart else 0,
        health_record.exercise_hours_week,
        health_record.alcohol_drinks_week
    ]).reshape(1, -1)
    
    return features

async def _log_prediction_for_fairness(
    model_type: str,
    features: np.ndarray,
    prediction: Dict[str, Any],
    health_record: HealthRecord
):
    """
    Log prediction for fairness monitoring
    """
    try:
        # Extract protected attributes
        protected_attrs = {
            'age_group': 'senior' if health_record.age >= 65 else ('middle' if health_record.age >= 40 else 'young'),
            'gender': health_record.gender,
            'race': health_record.race or 'unknown'
        }
        
        # Log prediction
        fairness_monitor.log_prediction(
            model_type=model_type,
            prediction=prediction,
            protected_attributes=protected_attrs
        )
        
    except Exception as e:
        logger.warning(f"Failed to log prediction for fairness monitoring: {str(e)}") 