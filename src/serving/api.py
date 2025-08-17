from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import asyncio
import uvicorn
from datetime import datetime
import numpy as np
import torch
import yaml
from pathlib import Path

from .cache import CacheManager
from .inference import RecommendationInference
from ..utils.logger import get_logger
from ..utils.config import Config, get_config_dir
from ..utils.metrics import get_metrics, APIMetrics


logger = get_logger(__name__)


# Pydantic models for API
class RecommendationRequest(BaseModel):
    """Request model for recommendations"""
    user_id: int = Field(..., description="User ID")
    num_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations")
    filter_categories: Optional[List[str]] = Field(None, description="Filter by categories")
    exclude_items: Optional[List[int]] = Field(None, description="Items to exclude")
    context: Optional[Dict[str, Any]] = Field(None, description="Contextual information")
    use_reranking: bool = Field(True, description="Whether to use CTR re-ranking")
    explain: bool = Field(False, description="Include explanation for recommendations")


class RecommendationItem(BaseModel):
    """Individual recommendation item"""
    item_id: int
    title: str
    score: float
    categories: Optional[List[str]] = None
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    user_id: int
    recommendations: List[RecommendationItem]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    models_loaded: bool
    cache_status: str
    version: str = "1.0.0"


class FeedbackRequest(BaseModel):
    """User feedback request"""
    user_id: int
    item_id: int
    action: str  # 'view', 'like', 'dislike', 'share', 'purchase'
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class MetricsResponse(BaseModel):
    """System metrics response"""
    cache_stats: Dict[str, Any]
    inference_stats: Dict[str, Any]
    system_stats: Dict[str, Any]


# Create FastAPI app
app = FastAPI(
    title="Product Recommendation Engine API",
    description="Production-grade recommendation system with two-tower architecture, BERT embeddings, and FAISS search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global variables for models and cache
cache_manager: Optional[CacheManager] = None
inference_engine: Optional[RecommendationInference] = None
config: Optional[Config] = None

# Metrics tracking
metrics = get_metrics()
api_metrics = APIMetrics(metrics)

# Performance tracking
request_times = []
request_count = 0


@app.on_event("startup")
async def startup_event():
    """Initialize models and cache on startup"""
    
    global cache_manager, inference_engine, config
    
    logger.info("Starting recommendation API...")
    
    # Start system metrics collection
    metrics.start_system_metrics_collection(interval=30.0)
    
    # Load configuration
    try:
        config_path = get_config_dir() / "serving_config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                serving_config = yaml.safe_load(f)
            logger.info("Loaded serving configuration")
        else:
            serving_config = {}
            logger.warning("No serving config found, using defaults")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        serving_config = {}
    
    # Initialize cache
    try:
        redis_config = serving_config.get('redis', {})
        cache_manager = CacheManager(
            redis_config=redis_config if redis_config else None,
            fallback_to_memory=True,
            memory_cache_size=1000
        )
        logger.info("Cache initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing cache: {e}")
        cache_manager = CacheManager(redis_config=None, fallback_to_memory=True)
    
    # Initialize inference engine
    try:
        model_config = serving_config.get('models', {})
        inference_engine = RecommendationInference(
            model_config=model_config,
            cache_manager=cache_manager
        )
        await inference_engine.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        inference_engine = None
    
    logger.info("API startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    
    global cache_manager, inference_engine
    
    logger.info("Shutting down recommendation API...")
    
    # Stop metrics collection
    metrics.stop_system_metrics_collection()
    
    if cache_manager and hasattr(cache_manager, 'redis_cache') and cache_manager.redis_cache:
        cache_manager.redis_cache.close()
    
    logger.info("API shutdown completed")


def get_cache_manager() -> CacheManager:
    """Dependency to get cache manager"""
    if cache_manager is None:
        raise HTTPException(status_code=500, detail="Cache not initialized")
    return cache_manager


def get_inference_engine() -> RecommendationInference:
    """Dependency to get inference engine"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    return inference_engine


@app.middleware("http")
async def track_performance(request: Request, call_next):
    """Middleware to track request performance"""
    
    global request_times, request_count
    
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Track performance
    request_times.append(process_time)
    request_count += 1
    
    # Keep only recent requests (last 1000)
    if len(request_times) > 1000:
        request_times = request_times[-1000:]
    
    # Record metrics
    api_metrics.record_request(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        duration=process_time
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Product Recommendation Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(
    cache: CacheManager = Depends(get_cache_manager)
):
    """Health check endpoint"""
    
    models_loaded = inference_engine is not None and inference_engine.models_loaded
    cache_stats = cache.get_stats()
    cache_status = cache_stats.get('status', 'unknown')
    
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        timestamp=datetime.now(),
        models_loaded=models_loaded,
        cache_status=cache_status
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    inference: RecommendationInference = Depends(get_inference_engine)
):
    """
    Get personalized recommendations for a user
    
    This endpoint provides the core recommendation functionality using:
    - Two-tower model for candidate generation
    - FAISS for fast similarity search  
    - LightGBM for CTR-based re-ranking
    - Redis caching for performance
    """
    
    start_time = time.time()
    
    try:
        # Generate recommendations
        recommendations = await inference.get_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            filter_categories=request.filter_categories,
            exclude_items=request.exclude_items or [],
            context=request.context or {},
            use_reranking=request.use_reranking,
            explain=request.explain
        )
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            metadata={
                "latency_ms": latency_ms,
                "cached": getattr(recommendations, '_cached', False),
                "model_version": "1.0.0",
                "num_candidates": len(recommendations),
                "reranked": request.use_reranking
            }
        )
        
        # Log performance
        logger.info(f"Recommendations for user {request.user_id}: {latency_ms:.2f}ms")
        
        # Track user interaction asynchronously
        background_tasks.add_task(
            track_recommendation_request,
            request.user_id,
            request.num_recommendations,
            latency_ms
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating recommendations for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )


@app.post("/feedback")
async def record_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """
    Record user feedback for model improvement
    
    Feedback is used to:
    - Update user profiles
    - Retrain models
    - A/B test evaluation
    """
    
    try:
        # Process feedback asynchronously
        background_tasks.add_task(
            process_user_feedback,
            feedback.user_id,
            feedback.item_id,
            feedback.action,
            feedback.context or {},
            feedback.timestamp or datetime.now()
        )
        
        return {"status": "feedback recorded", "user_id": feedback.user_id}
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail="Error recording feedback")


@app.get("/user/{user_id}/profile")
async def get_user_profile(
    user_id: int,
    inference: RecommendationInference = Depends(get_inference_engine)
):
    """Get user profile information"""
    
    try:
        profile = await inference.get_user_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving user profile")


@app.post("/user/{user_id}/invalidate-cache")
async def invalidate_user_cache(
    user_id: int,
    cache: CacheManager = Depends(get_cache_manager)
):
    """Invalidate all cache entries for a user"""
    
    try:
        if hasattr(cache, 'redis_cache') and cache.redis_cache:
            deleted_count = cache.redis_cache.invalidate_user_cache(user_id)
        else:
            deleted_count = 0
        
        return {
            "status": "cache invalidated",
            "user_id": user_id,
            "deleted_entries": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error invalidating cache for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error invalidating cache")


@app.get("/similar-items/{item_id}")
async def get_similar_items(
    item_id: int,
    num_items: int = 10,
    inference: RecommendationInference = Depends(get_inference_engine)
):
    """Get items similar to a given item"""
    
    try:
        similar_items = await inference.get_similar_items(item_id, num_items)
        
        return {
            "item_id": item_id,
            "similar_items": similar_items
        }
        
    except Exception as e:
        logger.error(f"Error getting similar items for {item_id}: {e}")
        raise HTTPException(status_code=500, detail="Error finding similar items")


@app.get("/trending")
async def get_trending_items(
    num_items: int = 20,
    category: Optional[str] = None,
    time_window: str = "24h",
    inference: RecommendationInference = Depends(get_inference_engine)
):
    """Get trending items"""
    
    try:
        trending = await inference.get_trending_items(
            num_items, category, time_window
        )
        
        return {
            "trending_items": trending,
            "category": category,
            "time_window": time_window
        }
        
    except Exception as e:
        logger.error(f"Error getting trending items: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving trending items")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics_json(
    cache: CacheManager = Depends(get_cache_manager)
):
    """Get system metrics and statistics in JSON format"""
    
    try:
        # Cache stats
        cache_stats = cache.get_stats()
        
        # Inference stats
        inference_stats = {}
        if inference_engine:
            inference_stats = inference_engine.get_stats()
        
        # System stats
        system_stats = {
            "total_requests": request_count,
            "avg_response_time_ms": np.mean(request_times[-100:]) * 1000 if request_times else 0,
            "p95_response_time_ms": np.percentile(request_times[-100:], 95) * 1000 if request_times else 0,
            "p99_response_time_ms": np.percentile(request_times[-100:], 99) * 1000 if request_times else 0,
            "qps": len([t for t in request_times[-100:] if time.time() - t < 1]) if request_times else 0
        }
        
        return MetricsResponse(
            cache_stats=cache_stats,
            inference_stats=inference_stats,
            system_stats=system_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving metrics")


@app.get("/metrics/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Get metrics in Prometheus text format"""
    
    try:
        return metrics.get_metrics_text()
        
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving Prometheus metrics")


@app.get("/model/status")
async def get_model_status():
    """Get model loading status and information"""
    
    if not inference_engine:
        return {"status": "not_loaded", "models": {}}
    
    return {
        "status": "loaded" if inference_engine.models_loaded else "loading",
        "models": inference_engine.get_model_info(),
        "loaded_at": getattr(inference_engine, 'loaded_at', None)
    }


# Background task functions
async def track_recommendation_request(
    user_id: int, 
    num_recommendations: int, 
    latency_ms: float
):
    """Track recommendation request for analytics"""
    
    # In a production system, this would write to a data warehouse
    # For now, we'll just log it
    logger.info(
        f"ANALYTICS: user_id={user_id}, num_recs={num_recommendations}, "
        f"latency_ms={latency_ms:.2f}, timestamp={datetime.now().isoformat()}"
    )


async def process_user_feedback(
    user_id: int,
    item_id: int,
    action: str,
    context: Dict[str, Any],
    timestamp: datetime
):
    """Process user feedback asynchronously"""
    
    # In production, this would:
    # 1. Update user profile in database
    # 2. Update item statistics
    # 3. Queue for model retraining
    # 4. Update A/B test metrics
    
    logger.info(
        f"FEEDBACK: user_id={user_id}, item_id={item_id}, action={action}, "
        f"timestamp={timestamp.isoformat()}"
    )
    
    # Invalidate user cache after feedback
    if cache_manager and hasattr(cache_manager, 'redis_cache') and cache_manager.redis_cache:
        cache_manager.redis_cache.invalidate_user_cache(user_id)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return {"error": "Not found", "detail": exc.detail, "path": request.url.path}


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}")
    return {"error": "Internal server error", "detail": "An unexpected error occurred"}


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )