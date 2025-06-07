"""
LifeLens: Predictive Health and Survival Insight System
Main FastAPI application entry point
"""
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import PlainTextResponse

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.routes import router as api_router
from app.core.model_manager import ModelManager

# Setup structured logging
setup_logging()
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('lifelens_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('lifelens_request_duration_seconds', 'Request duration')

# Global model manager instance
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    logger.info("Starting LifeLens API server")
    
    # Initialize model manager
    global model_manager
    model_manager = ModelManager()
    await model_manager.initialize()
    
    # Add model manager to app state
    app.state.model_manager = model_manager
    
    logger.info("LifeLens API server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LifeLens API server")
    if model_manager:
        await model_manager.cleanup()
    logger.info("LifeLens API server shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="LifeLens API",
    description="Predictive Health and Survival Insight System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect request metrics."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        logger.error("Request failed", error=str(e), path=request.url.path)
        status_code = 500
        raise
    finally:
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(status_code)
        ).inc()
        
        REQUEST_DURATION.observe(time.time() - start_time)
    
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Middleware for request/response logging."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else "unknown"
    )
    
    try:
        response = await call_next(request)
        
        # Log successful response
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=round(time.time() - start_time, 3)
        )
        
        return response
    
    except Exception as e:
        # Log error
        logger.error(
            "Request failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            duration=round(time.time() - start_time, 3)
        )
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unhandled errors."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path)
        }
    )


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time(),
        "models_loaded": model_manager is not None and model_manager.is_initialized()
    }


# Metrics endpoint for Prometheus
@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


# Include API routes
app.include_router(api_router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to LifeLens API",
        "description": "Predictive Health and Survival Insight System",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 