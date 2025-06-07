"""
Fairness Monitor for LifeLens
Monitors ML predictions for bias and fairness across protected attributes
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings

from app.core.config import settings
from app.core.logging import LoggerMixin


class FairnessMonitor(LoggerMixin):
    """
    Monitor ML model predictions for fairness and bias.
    Tracks predictions across protected attributes and detects potential bias.
    """
    
    def __init__(self):
        self.protected_attributes = settings.PROTECTED_ATTRIBUTES
        self.fairness_threshold = settings.FAIRNESS_THRESHOLD
        self.prediction_history = []
        self.bias_alerts = []
        self.fairness_metrics = {}
        
        # Bias detection thresholds
        self.thresholds = {
            'demographic_parity': 0.1,  # Max difference in positive rates
            'equalized_odds': 0.1,      # Max difference in TPR/FPR
            'calibration': 0.1          # Max difference in calibration
        }
    
    async def check_survival_prediction(
        self, 
        patient_data: Dict[str, Any], 
        prediction: Dict[str, float]
    ) -> None:
        """Check survival prediction for fairness issues."""
        
        # Extract protected attributes
        protected_attrs = self._extract_protected_attributes(patient_data)
        
        # Store prediction for analysis
        self.prediction_history.append({
            'timestamp': datetime.utcnow(),
            'prediction_type': 'survival',
            'prediction': prediction,
            'protected_attributes': protected_attrs,
            'patient_id': patient_data.get('patient_id', 'unknown')
        })
        
        # Check for immediate bias issues
        await self._check_demographic_parity('survival', protected_attrs, prediction)
        
        # Periodic comprehensive analysis
        if len(self.prediction_history) % 100 == 0:
            await self._comprehensive_fairness_analysis()
    
    async def check_event_prediction(
        self, 
        patient_data: Dict[str, Any], 
        predictions: Dict[str, float]
    ) -> None:
        """Check event prediction for fairness issues."""
        
        protected_attrs = self._extract_protected_attributes(patient_data)
        
        # Store prediction for analysis
        self.prediction_history.append({
            'timestamp': datetime.utcnow(),
            'prediction_type': 'event',
            'prediction': predictions,
            'protected_attributes': protected_attrs,
            'patient_id': patient_data.get('patient_id', 'unknown')
        })
        
        # Check each event type for bias
        for event_type, probability in predictions.items():
            await self._check_demographic_parity(
                f'event_{event_type}', 
                protected_attrs, 
                {event_type: probability}
            )
        
        # Periodic analysis
        if len(self.prediction_history) % 100 == 0:
            await self._comprehensive_fairness_analysis()
    
    def _extract_protected_attributes(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract protected attributes from patient data."""
        protected_attrs = {}
        
        # Age groups
        age = patient_data.get('age', 0)
        if age < 30:
            protected_attrs['age_group'] = 'young'
        elif age < 65:
            protected_attrs['age_group'] = 'middle'
        else:
            protected_attrs['age_group'] = 'senior'
        
        # Gender
        gender = patient_data.get('gender', 'unknown')
        if gender == 1 or gender == 'male':
            protected_attrs['gender'] = 'male'
        elif gender == 0 or gender == 'female':
            protected_attrs['gender'] = 'female'
        else:
            protected_attrs['gender'] = 'unknown'
        
        # Race (if available)
        race = patient_data.get('race', 'unknown')
        if isinstance(race, (int, float)):
            race_mapping = {0: 'white', 1: 'black', 2: 'hispanic', 3: 'asian', 4: 'other'}
            protected_attrs['race'] = race_mapping.get(race, 'unknown')
        else:
            protected_attrs['race'] = str(race)
        
        # Ethnicity (if available)
        ethnicity = patient_data.get('ethnicity', 'unknown')
        if isinstance(ethnicity, (int, float)):
            ethnicity_mapping = {0: 'non_hispanic', 1: 'hispanic'}
            protected_attrs['ethnicity'] = ethnicity_mapping.get(ethnicity, 'unknown')
        else:
            protected_attrs['ethnicity'] = str(ethnicity)
        
        return protected_attrs
    
    async def _check_demographic_parity(
        self, 
        prediction_type: str, 
        protected_attrs: Dict[str, Any], 
        prediction: Dict[str, float]
    ) -> None:
        """Check for demographic parity violations."""
        
        # Get recent predictions of the same type
        recent_predictions = [
            p for p in self.prediction_history[-100:]  # Last 100 predictions
            if p['prediction_type'] == prediction_type.split('_')[0]
        ]
        
        if len(recent_predictions) < 20:  # Need minimum samples
            return
        
        # Analyze by each protected attribute
        for attr_name, attr_value in protected_attrs.items():
            if attr_name not in self.protected_attributes:
                continue
            
            # Group predictions by attribute value
            groups = {}
            for pred in recent_predictions:
                pred_attrs = pred['protected_attributes']
                group_value = pred_attrs.get(attr_name, 'unknown')
                
                if group_value not in groups:
                    groups[group_value] = []
                
                # Extract relevant prediction value
                if prediction_type == 'survival':
                    pred_value = pred['prediction'].get('5_year', 0.5)
                else:
                    event_name = prediction_type.replace('event_', '')
                    pred_value = pred['prediction'].get(event_name, 0.0)
                
                groups[group_value].append(pred_value)
            
            # Check for significant differences between groups
            if len(groups) >= 2:
                await self._analyze_group_differences(
                    prediction_type, attr_name, groups
                )
    
    async def _analyze_group_differences(
        self, 
        prediction_type: str, 
        attr_name: str, 
        groups: Dict[str, List[float]]
    ) -> None:
        """Analyze differences between demographic groups."""
        
        group_stats = {}
        for group_name, predictions in groups.items():
            if len(predictions) >= 5:  # Minimum sample size
                group_stats[group_name] = {
                    'mean': np.mean(predictions),
                    'count': len(predictions),
                    'std': np.std(predictions)
                }
        
        if len(group_stats) < 2:
            return
        
        # Find max and min group means
        group_means = {name: stats['mean'] for name, stats in group_stats.items()}
        max_group = max(group_means, key=group_means.get)
        min_group = min(group_means, key=group_means.get)
        
        # Calculate difference
        difference = group_means[max_group] - group_means[min_group]
        
        # Check if difference exceeds threshold
        if difference > self.thresholds['demographic_parity']:
            await self._raise_bias_alert(
                prediction_type, attr_name, max_group, min_group, 
                difference, group_stats
            )
    
    async def _raise_bias_alert(
        self, 
        prediction_type: str, 
        attr_name: str, 
        advantaged_group: str, 
        disadvantaged_group: str,
        difference: float, 
        group_stats: Dict[str, Dict[str, float]]
    ) -> None:
        """Raise a bias alert when unfairness is detected."""
        
        alert = {
            'timestamp': datetime.utcnow(),
            'alert_type': 'demographic_parity_violation',
            'prediction_type': prediction_type,
            'protected_attribute': attr_name,
            'advantaged_group': advantaged_group,
            'disadvantaged_group': disadvantaged_group,
            'difference': difference,
            'threshold': self.thresholds['demographic_parity'],
            'group_statistics': group_stats,
            'severity': 'high' if difference > 0.2 else 'medium'
        }
        
        self.bias_alerts.append(alert)
        
        # Log the alert
        self.logger.error(
            "Bias alert raised",
            prediction_type=prediction_type,
            protected_attribute=attr_name,
            difference=difference,
            advantaged_group=advantaged_group,
            disadvantaged_group=disadvantaged_group
        )
        
        # Store in fairness metrics
        metric_key = f"{prediction_type}_{attr_name}"
        if metric_key not in self.fairness_metrics:
            self.fairness_metrics[metric_key] = []
        
        self.fairness_metrics[metric_key].append({
            'timestamp': alert['timestamp'],
            'difference': difference,
            'violation': True
        })
    
    async def _comprehensive_fairness_analysis(self) -> None:
        """Perform comprehensive fairness analysis on all stored predictions."""
        
        self.logger.info("Performing comprehensive fairness analysis")
        
        if len(self.prediction_history) < 50:
            return
        
        # Analyze by prediction type
        prediction_types = set(p['prediction_type'] for p in self.prediction_history)
        
        for pred_type in prediction_types:
            type_predictions = [
                p for p in self.prediction_history 
                if p['prediction_type'] == pred_type
            ]
            
            await self._analyze_prediction_type_fairness(pred_type, type_predictions)
    
    async def _analyze_prediction_type_fairness(
        self, 
        pred_type: str, 
        predictions: List[Dict[str, Any]]
    ) -> None:
        """Analyze fairness for a specific prediction type."""
        
        # Convert to DataFrame for easier analysis
        data_rows = []
        for pred in predictions:
            row = {
                'timestamp': pred['timestamp'],
                'patient_id': pred['patient_id']
            }
            
            # Add protected attributes
            row.update(pred['protected_attributes'])
            
            # Add prediction values
            if pred_type == 'survival':
                row['survival_5_year'] = pred['prediction'].get('5_year', 0.5)
                row['survival_10_year'] = pred['prediction'].get('10_year', 0.5)
            else:  # event predictions
                for event_name, prob in pred['prediction'].items():
                    row[f'event_{event_name}'] = prob
            
            data_rows.append(row)
        
        if not data_rows:
            return
        
        df = pd.DataFrame(data_rows)
        
        # Analyze each protected attribute
        for attr in self.protected_attributes:
            if attr in df.columns:
                await self._analyze_attribute_fairness(df, attr, pred_type)
    
    async def _analyze_attribute_fairness(
        self, 
        df: pd.DataFrame, 
        attr: str, 
        pred_type: str
    ) -> None:
        """Analyze fairness for a specific protected attribute."""
        
        # Get prediction columns
        pred_cols = [col for col in df.columns if col.startswith(('survival_', 'event_'))]
        
        for pred_col in pred_cols:
            # Group by attribute value
            grouped = df.groupby(attr)[pred_col].agg(['mean', 'count', 'std']).reset_index()
            
            if len(grouped) < 2:
                continue
            
            # Calculate statistical parity difference
            max_mean = grouped['mean'].max()
            min_mean = grouped['mean'].min()
            parity_diff = max_mean - min_mean
            
            # Store metric
            metric_key = f"{pred_type}_{pred_col}_{attr}"
            if metric_key not in self.fairness_metrics:
                self.fairness_metrics[metric_key] = []
            
            self.fairness_metrics[metric_key].append({
                'timestamp': datetime.utcnow(),
                'statistical_parity_difference': parity_diff,
                'group_stats': grouped.to_dict('records'),
                'violation': parity_diff > self.thresholds['demographic_parity']
            })
    
    def get_fairness_report(self) -> Dict[str, Any]:
        """Generate comprehensive fairness report."""
        
        # Count alerts by type and attribute
        alert_summary = {}
        for alert in self.bias_alerts:
            key = f"{alert['protected_attribute']}_{alert['prediction_type']}"
            if key not in alert_summary:
                alert_summary[key] = 0
            alert_summary[key] += 1
        
        # Recent fairness status
        recent_violations = len([
            alert for alert in self.bias_alerts 
            if (datetime.utcnow() - alert['timestamp']).days <= 7
        ])
        
        # Overall fairness score
        total_metrics = sum(len(metrics) for metrics in self.fairness_metrics.values())
        total_violations = sum(
            1 for metrics in self.fairness_metrics.values()
            for metric in metrics
            if metric.get('violation', False)
        )
        
        fairness_score = 1.0 - (total_violations / max(total_metrics, 1))
        
        return {
            'fairness_score': fairness_score,
            'total_predictions_analyzed': len(self.prediction_history),
            'total_bias_alerts': len(self.bias_alerts),
            'recent_violations_7_days': recent_violations,
            'alert_summary': alert_summary,
            'protected_attributes_monitored': self.protected_attributes,
            'fairness_thresholds': self.thresholds,
            'last_analysis': datetime.utcnow().isoformat(),
            'status': 'healthy' if fairness_score > 0.8 else 'needs_attention'
        }
    
    def get_recent_alerts(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent bias alerts."""
        cutoff_date = datetime.utcnow() - pd.Timedelta(days=days)
        
        return [
            alert for alert in self.bias_alerts
            if alert['timestamp'] > cutoff_date
        ]
    
    def clear_history(self, keep_days: int = 30) -> None:
        """Clear old prediction history to manage memory."""
        cutoff_date = datetime.utcnow() - pd.Timedelta(days=keep_days)
        
        # Keep recent predictions
        self.prediction_history = [
            pred for pred in self.prediction_history
            if pred['timestamp'] > cutoff_date
        ]
        
        # Keep recent alerts
        self.bias_alerts = [
            alert for alert in self.bias_alerts
            if alert['timestamp'] > cutoff_date
        ]
        
        self.logger.info(f"Cleared fairness history older than {keep_days} days") 