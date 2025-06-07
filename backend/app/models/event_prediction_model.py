"""
Event Prediction Model for LifeLens
Predicts health events using ensemble of XGBoost, LightGBM, and Neural Networks
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings

# ML libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.neural_network import MLPClassifier

from app.models.base_model import BaseModel, MLModelMixin
from app.core.config import settings


class EventPredictionModel(BaseModel, MLModelMixin):
    """
    Advanced health event prediction model using ensemble methods.
    Predicts probabilities for various health events like cardiac arrest, stroke, etc.
    """
    
    def __init__(self):
        super().__init__("event_prediction_model")
        MLModelMixin.__init__(self)
        
        # Model components
        self.xgb_models = {}
        self.lgb_models = {}
        self.rf_models = {}
        self.nn_models = {}
        self.scaler = StandardScaler()
        
        # Feature configuration
        self.feature_names = [
            # Demographics
            "age", "gender", "bmi", "race", "ethnicity",
            
            # Vital signs
            "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
            "temperature", "oxygen_saturation",
            
            # Lab values
            "cholesterol_total", "cholesterol_hdl", "cholesterol_ldl",
            "triglycerides", "glucose_fasting", "hba1c",
            "creatinine", "bun", "hemoglobin", "hematocrit",
            "white_blood_cells", "platelets", "sodium", "potassium",
            
            # Lifestyle factors
            "smoking_status", "alcohol_consumption", "physical_activity",
            "diet_quality", "sleep_hours", "stress_level",
            
            # Medical history
            "diabetes", "hypertension", "heart_disease", "stroke_history",
            "kidney_disease", "liver_disease", "cancer_history",
            "family_history_chd", "family_history_diabetes",
            
            # Medications
            "ace_inhibitors", "beta_blockers", "statins", "diuretics",
            "anticoagulants", "insulin", "metformin", "medications_count",
            
            # Healthcare utilization
            "hospital_admissions_last_year", "emergency_visits_last_year",
            "specialist_visits", "preventive_care_visits"
        ]
        
        # Event types to predict
        self.event_types = [
            "cardiac_arrest", "stroke", "diabetes_onset",
            "heart_failure", "kidney_disease", "liver_disease"
        ]
        
        # Model ensemble weights
        self.ensemble_weights = {
            "xgboost": 0.35,
            "lightgbm": 0.35,
            "random_forest": 0.20,
            "neural_network": 0.10
        }
        
        # Performance metrics
        self.auc_scores = {}
        self.feature_importance_scores = {}
    
    async def initialize(self) -> None:
        """Initialize the event prediction model with synthetic training data."""
        self.logger.info("Initializing event prediction model")
        
        try:
            # Generate synthetic training data
            synthetic_data = self._generate_synthetic_event_data(n_samples=10000)
            
            # Train models for each event type
            await self._train_models(synthetic_data)
            
            self.mark_as_trained()
            self.logger.info("Event prediction model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize event prediction model: {str(e)}")
            raise
    
    def _generate_synthetic_event_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic health event data for training."""
        np.random.seed(settings.RANDOM_STATE)
        
        # Demographics
        data = {
            "age": np.random.normal(55, 20, n_samples).clip(18, 100),
            "gender": np.random.choice([0, 1], n_samples),  # 0: female, 1: male
            "bmi": np.random.normal(28, 6, n_samples).clip(15, 50),
            "race": np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
            "ethnicity": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        }
        
        # Vital signs
        data.update({
            "systolic_bp": np.random.normal(130, 25, n_samples).clip(80, 220),
            "diastolic_bp": np.random.normal(80, 15, n_samples).clip(50, 140),
            "heart_rate": np.random.normal(75, 15, n_samples).clip(50, 120),
            "respiratory_rate": np.random.normal(16, 4, n_samples).clip(10, 30),
            "temperature": np.random.normal(98.6, 1, n_samples).clip(96, 104),
            "oxygen_saturation": np.random.normal(98, 2, n_samples).clip(85, 100),
        })
        
        # Lab values
        data.update({
            "cholesterol_total": np.random.normal(200, 50, n_samples).clip(100, 400),
            "cholesterol_hdl": np.random.normal(50, 15, n_samples).clip(20, 100),
            "cholesterol_ldl": np.random.normal(120, 35, n_samples).clip(50, 250),
            "triglycerides": np.random.normal(150, 75, n_samples).clip(50, 500),
            "glucose_fasting": np.random.normal(100, 30, n_samples).clip(70, 400),
            "hba1c": np.random.normal(5.5, 1.5, n_samples).clip(4, 15),
            "creatinine": np.random.normal(1.0, 0.5, n_samples).clip(0.5, 5),
            "bun": np.random.normal(15, 8, n_samples).clip(5, 50),
            "hemoglobin": np.random.normal(14, 2.5, n_samples).clip(8, 20),
            "hematocrit": np.random.normal(42, 6, n_samples).clip(25, 60),
            "white_blood_cells": np.random.normal(7, 3, n_samples).clip(2, 20),
            "platelets": np.random.normal(250, 75, n_samples).clip(50, 500),
            "sodium": np.random.normal(140, 5, n_samples).clip(130, 150),
            "potassium": np.random.normal(4, 0.8, n_samples).clip(2.5, 6),
        })
        
        # Lifestyle factors
        data.update({
            "smoking_status": np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15]),
            "alcohol_consumption": np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
            "physical_activity": np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
            "diet_quality": np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),
            "sleep_hours": np.random.normal(7, 1.5, n_samples).clip(4, 12),
            "stress_level": np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
        })
        
        # Medical history
        data.update({
            "diabetes": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            "hypertension": np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
            "heart_disease": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            "stroke_history": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            "kidney_disease": np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
            "liver_disease": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            "cancer_history": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            "family_history_chd": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            "family_history_diabetes": np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        })
        
        # Medications
        data.update({
            "ace_inhibitors": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            "beta_blockers": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            "statins": np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
            "diuretics": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            "anticoagulants": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            "insulin": np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            "metformin": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            "medications_count": np.random.poisson(4, n_samples).clip(0, 20),
        })
        
        # Healthcare utilization
        data.update({
            "hospital_admissions_last_year": np.random.poisson(0.5, n_samples).clip(0, 10),
            "emergency_visits_last_year": np.random.poisson(1, n_samples).clip(0, 15),
            "specialist_visits": np.random.poisson(2, n_samples).clip(0, 20),
            "preventive_care_visits": np.random.poisson(1.5, n_samples).clip(0, 10),
        })
        
        df = pd.DataFrame(data)
        
        # Generate event outcomes based on risk factors
        for event in self.event_types:
            df[event] = self._generate_event_outcome(df, event)
        
        return df
    
    def _generate_event_outcome(self, df: pd.DataFrame, event_type: str) -> np.ndarray:
        """Generate synthetic event outcomes based on risk factors."""
        # Define risk weights for different events
        risk_weights = {
            "cardiac_arrest": {
                "age": 0.08, "gender": 0.5, "systolic_bp": 0.01, "heart_disease": 2.5,
                "smoking_status": 0.8, "diabetes": 1.2, "family_history_chd": 0.7
            },
            "stroke": {
                "age": 0.06, "systolic_bp": 0.015, "diabetes": 1.0, "heart_disease": 1.5,
                "smoking_status": 0.6, "hypertension": 1.8, "stroke_history": 3.0
            },
            "diabetes_onset": {
                "age": 0.04, "bmi": 0.08, "glucose_fasting": 0.02, "family_history_diabetes": 1.5,
                "physical_activity": -0.5, "diet_quality": -0.3
            },
            "heart_failure": {
                "age": 0.05, "systolic_bp": 0.01, "heart_disease": 2.0, "diabetes": 0.8,
                "hypertension": 1.2, "kidney_disease": 1.5
            },
            "kidney_disease": {
                "age": 0.03, "diabetes": 1.8, "hypertension": 1.5, "creatinine": 0.5,
                "heart_disease": 0.8
            },
            "liver_disease": {
                "age": 0.02, "alcohol_consumption": 1.5, "bmi": 0.05, "diabetes": 0.6
            }
        }
        
        weights = risk_weights.get(event_type, {})
        
        # Calculate risk score
        risk_score = 0
        for feature, weight in weights.items():
            if feature in df.columns:
                risk_score += weight * df[feature]
        
        # Convert to probability using logistic function
        base_rate = {"cardiac_arrest": 0.01, "stroke": 0.02, "diabetes_onset": 0.05,
                    "heart_failure": 0.03, "kidney_disease": 0.04, "liver_disease": 0.02}
        
        prob = base_rate.get(event_type, 0.02) * np.exp(risk_score)
        prob = prob / (1 + prob)  # Logistic transformation
        
        # Generate binary outcomes
        return np.random.binomial(1, prob.clip(0, 1), len(df))
    
    async def _train_models(self, data: pd.DataFrame) -> None:
        """Train ensemble models for each event type."""
        self.logger.info("Training event prediction models")
        
        # Prepare features
        X = data[self.feature_names].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models for each event type
        for event_type in self.event_types:
            self.logger.info(f"Training models for {event_type}")
            
            y = data[event_type].values
            
            # Train individual models
            await self._train_xgboost(X, y, event_type)
            await self._train_lightgbm(X, y, event_type)
            await self._train_random_forest(X, y, event_type)
            await self._train_neural_network(X_scaled, y, event_type)
            
            # Evaluate ensemble performance
            self._evaluate_ensemble(X_scaled, y, event_type)
    
    async def _train_xgboost(self, X: pd.DataFrame, y: np.ndarray, event_type: str) -> None:
        """Train XGBoost model for specific event type."""
        if event_type not in self.xgb_models:
            self.xgb_models[event_type] = {}
        
        # Handle class imbalance
        scale_pos_weight = len(y[y == 0]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 1
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=settings.RANDOM_STATE,
            eval_metric='auc'
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        
        self.xgb_models[event_type] = model
        
        # Store feature importance
        if event_type not in self.feature_importance_scores:
            self.feature_importance_scores[event_type] = {}
        self.feature_importance_scores[event_type]['xgboost'] = model.feature_importances_
    
    async def _train_lightgbm(self, X: pd.DataFrame, y: np.ndarray, event_type: str) -> None:
        """Train LightGBM model for specific event type."""
        if event_type not in self.lgb_models:
            self.lgb_models[event_type] = {}
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=settings.RANDOM_STATE,
            verbosity=-1
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        
        self.lgb_models[event_type] = model
        self.feature_importance_scores[event_type]['lightgbm'] = model.feature_importances_
    
    async def _train_random_forest(self, X: pd.DataFrame, y: np.ndarray, event_type: str) -> None:
        """Train Random Forest model for specific event type."""
        if event_type not in self.rf_models:
            self.rf_models[event_type] = {}
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=settings.RANDOM_STATE,
            class_weight='balanced'
        )
        
        model.fit(X, y)
        self.rf_models[event_type] = model
        self.feature_importance_scores[event_type]['random_forest'] = model.feature_importances_
    
    async def _train_neural_network(self, X: np.ndarray, y: np.ndarray, event_type: str) -> None:
        """Train Neural Network model for specific event type."""
        if event_type not in self.nn_models:
            self.nn_models[event_type] = {}
        
        model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=settings.RANDOM_STATE,
            early_stopping=True,
            validation_fraction=0.2
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
        
        self.nn_models[event_type] = model
    
    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray, event_type: str) -> None:
        """Evaluate ensemble performance for specific event type."""
        try:
            # Make predictions with each model
            predictions = self._get_ensemble_predictions(X, event_type)
            
            # Calculate AUC score
            if len(np.unique(y)) > 1:  # Check if both classes are present
                auc_score = roc_auc_score(y, predictions)
                self.auc_scores[event_type] = auc_score
                self.logger.info(f"{event_type} AUC: {auc_score:.3f}")
            else:
                self.auc_scores[event_type] = 0.5
                
        except Exception as e:
            self.logger.warning(f"Evaluation failed for {event_type}: {str(e)}")
            self.auc_scores[event_type] = 0.5
    
    def _get_ensemble_predictions(self, X: np.ndarray, event_type: str) -> np.ndarray:
        """Get ensemble predictions for specific event type."""
        predictions = np.zeros(len(X))
        
        # XGBoost predictions
        if event_type in self.xgb_models and self.xgb_models[event_type] is not None:
            xgb_pred = self.xgb_models[event_type].predict_proba(X)[:, 1]
            predictions += self.ensemble_weights["xgboost"] * xgb_pred
        
        # LightGBM predictions
        if event_type in self.lgb_models and self.lgb_models[event_type] is not None:
            lgb_pred = self.lgb_models[event_type].predict_proba(X)[:, 1]
            predictions += self.ensemble_weights["lightgbm"] * lgb_pred
        
        # Random Forest predictions
        if event_type in self.rf_models and self.rf_models[event_type] is not None:
            rf_pred = self.rf_models[event_type].predict_proba(X)[:, 1]
            predictions += self.ensemble_weights["random_forest"] * rf_pred
        
        # Neural Network predictions
        if event_type in self.nn_models and self.nn_models[event_type] is not None:
            nn_pred = self.nn_models[event_type].predict_proba(X)[:, 1]
            predictions += self.ensemble_weights["neural_network"] * nn_pred
        
        return predictions
    
    async def predict(self, input_data: np.ndarray, event_types: List[str] = None) -> Dict[str, float]:
        """Predict event probabilities for given input data."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        event_types = event_types or self.event_types
        
        # Validate input
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")
        
        # Scale features
        X_scaled = self.scaler.transform(input_data)
        
        # Get predictions for each event type
        predictions = {}
        for event_type in event_types:
            if event_type in self.event_types:
                pred = self._get_ensemble_predictions(X_scaled, event_type)
                predictions[event_type] = float(pred[0]) if len(pred) > 0 else 0.0
            else:
                predictions[event_type] = 0.0
        
        return predictions
    
    def get_feature_importance(self, event_type: str = None) -> Dict[str, Any]:
        """Get feature importance for specific event type or all events."""
        if event_type and event_type in self.feature_importance_scores:
            # Average importance across models for specific event
            importances = self.feature_importance_scores[event_type]
            avg_importance = np.mean([importances.get(model, np.zeros(len(self.feature_names))) 
                                    for model in importances.keys()], axis=0)
            
            return dict(zip(self.feature_names, avg_importance))
        else:
            # Return importance for all events
            return self.feature_importance_scores
    
    def _create_test_input(self) -> np.ndarray:
        """Create test input for health checks."""
        # Create a typical patient profile
        test_data = np.array([[
            55, 0, 28, 1, 0,  # demographics
            130, 85, 75, 16, 98.6, 98,  # vitals
            200, 45, 130, 150, 100, 5.5, 1.0, 15, 13, 40, 7, 250, 140, 4.0,  # labs
            1, 1, 1, 1, 7, 1,  # lifestyle
            0, 1, 0, 0, 0, 0, 0, 1, 0,  # medical history
            1, 0, 1, 0, 0, 0, 0, 3,  # medications
            0, 1, 2, 1  # healthcare utilization
        ]])
        return test_data
    
    def get_state(self) -> Dict[str, Any]:
        """Get model state for serialization."""
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "training_metadata": self.training_metadata,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            "event_types": self.event_types,
            "ensemble_weights": self.ensemble_weights,
            "auc_scores": self.auc_scores,
            "performance_metrics": self.performance_metrics
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load model state from serialization."""
        self.model_name = state["model_name"]
        self.model_id = state["model_id"]
        self.version = state["version"]
        self.created_at = datetime.fromisoformat(state["created_at"])
        if state["last_updated"]:
            self.last_updated = datetime.fromisoformat(state["last_updated"])
        self.training_metadata = state["training_metadata"]
        self.is_trained = state["is_trained"]
        self.feature_names = state["feature_names"]
        self.event_types = state["event_types"]
        self.ensemble_weights = state["ensemble_weights"]
        self.auc_scores = state["auc_scores"]
        self.performance_metrics = state["performance_metrics"] 