"""
Structured logging configuration for LifeLens
"""
import logging
import sys
from typing import Any, Dict, Optional
import structlog
from structlog.types import EventDict, Processor
from app.core.config import settings


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to log entries."""
    event_dict["app"] = settings.APP_NAME
    event_dict["version"] = settings.APP_VERSION
    return event_dict


def add_correlation_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add correlation ID if available from context."""
    # This would be populated by middleware with request ID
    # For now, we'll use a placeholder
    if "correlation_id" not in event_dict:
        event_dict["correlation_id"] = "unknown"
    return event_dict


def censor_sensitive_data(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Remove or mask sensitive data from logs."""
    sensitive_keys = {
        "password", "token", "secret", "key", "api_key", 
        "authorization", "cookie", "ssn", "credit_card"
    }
    
    def _censor_dict(data: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: _censor_dict(v) if k.lower() not in sensitive_keys 
                else "***REDACTED***" 
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [_censor_dict(item) for item in data]
        elif isinstance(data, str) and len(data) > 100:
            # Truncate very long strings (potential data dumps)
            return data[:97] + "..."
        return data
    
    return _censor_dict(event_dict)


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure stdlib logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    )
    
    # Configure processors chain
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        add_app_context,
        add_correlation_id,
        censor_sensitive_data,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="ISO", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if settings.LOG_FORMAT.lower() == "json":
        # JSON logging for production
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ])
    else:
        # Human-readable logging for development
        processors.extend([
            structlog.processors.ExceptionPrettyPrinter(),
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Configure loggers for external libraries
    configure_external_loggers()


def configure_external_loggers() -> None:
    """Configure logging levels for external libraries."""
    external_loggers = {
        "uvicorn.access": logging.WARNING,
        "uvicorn.error": logging.INFO,
        "fastapi": logging.INFO,
        "sqlalchemy.engine": logging.WARNING,
        "httpx": logging.WARNING,
        "transformers": logging.ERROR,
        "torch": logging.ERROR,
        "sklearn": logging.WARNING,
    }
    
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add structured logging to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger instance bound to this class."""
        return structlog.get_logger(self.__class__.__name__)


# Health monitoring logger
class HealthLogger:
    """Specialized logger for health monitoring and alerts."""
    
    def __init__(self):
        self.logger = structlog.get_logger("health_monitor")
    
    def log_prediction(
        self,
        patient_id: str,
        prediction_type: str,
        confidence: float,
        features_used: int,
        processing_time: float
    ) -> None:
        """Log prediction events."""
        self.logger.info(
            "Prediction completed",
            patient_id=patient_id,
            prediction_type=prediction_type,
            confidence=confidence,
            features_used=features_used,
            processing_time_ms=round(processing_time * 1000, 2)
        )
    
    def log_model_performance(
        self,
        model_name: str,
        metric_name: str,
        metric_value: float,
        threshold: Optional[float] = None
    ) -> None:
        """Log model performance metrics."""
        level = "info"
        if threshold and metric_value < threshold:
            level = "warning"
        
        getattr(self.logger, level)(
            "Model performance metric",
            model_name=model_name,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            status="below_threshold" if threshold and metric_value < threshold else "normal"
        )
    
    def log_data_quality_issue(
        self,
        dataset_name: str,
        issue_type: str,
        severity: str,
        affected_records: int,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log data quality issues."""
        self.logger.warning(
            "Data quality issue detected",
            dataset_name=dataset_name,
            issue_type=issue_type,
            severity=severity,
            affected_records=affected_records,
            details=details or {}
        )
    
    def log_fairness_violation(
        self,
        model_name: str,
        protected_attribute: str,
        metric_name: str,
        metric_value: float,
        threshold: float
    ) -> None:
        """Log fairness violations."""
        self.logger.error(
            "Fairness violation detected",
            model_name=model_name,
            protected_attribute=protected_attribute,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            violation_severity=abs(metric_value - threshold)
        )
    
    def log_extreme_environment_prediction(
        self,
        environment_type: str,
        subject_id: str,
        risk_level: str,
        environmental_factors: Dict[str, Any],
        recommendations: list[str]
    ) -> None:
        """Log extreme environment predictions."""
        self.logger.info(
            "Extreme environment prediction",
            environment_type=environment_type,
            subject_id=subject_id,
            risk_level=risk_level,
            environmental_factors=environmental_factors,
            recommendations=recommendations
        )


# Global health logger instance
health_logger = HealthLogger() 