"""
Configuration management for LifeLens API
"""
import os
from typing import List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import validator
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application settings
    APP_NAME: str = "LifeLens"
    APP_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "*"]
    
    # Database settings
    DATABASE_URL: Optional[str] = "sqlite:///./lifelens.db"
    POSTGRES_SERVER: Optional[str] = None
    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    POSTGRES_PORT: Optional[str] = "5432"
    
    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> str:
        """Assemble database URL from components if not provided directly."""
        if isinstance(v, str) and v:
            return v
        
        # Build PostgreSQL URL if components are provided
        postgres_server = values.get("POSTGRES_SERVER")
        if postgres_server:
            postgres_user = values.get("POSTGRES_USER", "postgres")
            postgres_password = values.get("POSTGRES_PASSWORD", "")
            postgres_db = values.get("POSTGRES_DB", "lifelens")
            postgres_port = values.get("POSTGRES_PORT", "5432")
            
            return f"postgresql://{postgres_user}:{postgres_password}@{postgres_server}:{postgres_port}/{postgres_db}"
        
        # Default to SQLite
        return "sqlite:///./lifelens.db"
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_EXPIRE_TIME: int = 3600  # 1 hour
    
    # Model settings
    MODEL_PATH: str = "./models"
    MODEL_CACHE_SIZE: int = 100
    MODEL_TIMEOUT: int = 30  # seconds
    
    # Data settings
    DATA_PATH: str = "./data"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_FILE_TYPES: List[str] = [".csv", ".json", ".parquet", ".txt"]
    
    # ML Pipeline settings
    BATCH_SIZE: int = 32
    MAX_FEATURES: int = 10000
    CROSS_VALIDATION_FOLDS: int = 5
    RANDOM_STATE: int = 42
    
    # Survival analysis settings
    SURVIVAL_HORIZONS: List[int] = [1, 5, 10]  # years
    CONFIDENCE_INTERVAL: float = 0.95
    
    # Health event prediction settings
    EVENT_TYPES: List[str] = [
        "cardiac_arrest",
        "stroke",
        "diabetes_onset",
        "heart_failure",
        "kidney_disease"
    ]
    
    # NLP settings
    NLP_MODEL_NAME: str = "clinical-bert"
    MAX_TEXT_LENGTH: int = 512
    NLP_BATCH_SIZE: int = 16
    
    # Time series settings
    TIME_SERIES_WINDOW: int = 30  # days
    SAMPLING_FREQUENCY: str = "1H"  # hourly
    
    # Fairness and bias settings
    PROTECTED_ATTRIBUTES: List[str] = ["age", "gender", "race", "ethnicity"]
    FAIRNESS_THRESHOLD: float = 0.8
    BIAS_MONITOR_INTERVAL: int = 24  # hours
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: Optional[str] = None
    
    # Monitoring settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Security settings
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # API rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    
    # External services
    MIMIC_API_URL: Optional[str] = None
    MIMIC_API_KEY: Optional[str] = None
    
    # Extreme environments settings (bonus features)
    ENABLE_ASTRONAUT_MODE: bool = False
    ENABLE_DEEP_SEA_MODE: bool = False
    
    # Astronaut-specific settings
    RADIATION_THRESHOLD: float = 100.0  # mSv
    MICROGRAVITY_ADAPTATION_DAYS: int = 14
    
    # Deep-sea specific settings
    PRESSURE_THRESHOLD: float = 10.0  # atmospheres
    DECOMPRESSION_SAFETY_FACTOR: float = 1.5
    
    class Config:
        """Pydantic configuration."""
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the full path for a model file."""
        return Path(self.MODEL_PATH) / f"{model_name}.pkl"
    
    def get_data_path(self, dataset_name: str) -> Path:
        """Get the full path for a dataset."""
        return Path(self.DATA_PATH) / dataset_name
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.DEBUG
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.DEBUG and os.getenv("ENVIRONMENT") == "production"


# Create global settings instance
settings = Settings()


# Environment-specific settings
class DevelopmentSettings(Settings):
    """Development environment settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    

class ProductionSettings(Settings):
    """Production environment settings."""
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-this-in-production")
    

class TestingSettings(Settings):
    """Testing environment settings."""
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///./test.db"
    REDIS_URL: str = "redis://localhost:6379/1"


def get_settings() -> Settings:
    """Get settings based on environment."""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings() 