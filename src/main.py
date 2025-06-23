# src/main.py - FastAPI MCP Server Entry Point
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime

# Import our modules
from .config import get_settings
from .vector.qdrant_client import QdrantClient
from .api import search, recommendations, admin
from .models.anime import DatabaseStats

# Get application settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# Global Qdrant client
qdrant_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global qdrant_client
    
    # Initialize Qdrant client with centralized configuration
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        settings=settings
    )
    
    # Health check with configured timeout
    if await qdrant_client.health_check():
        logger.info("✅ Qdrant connection established", extra={
            "url": settings.qdrant_url,
            "collection": settings.qdrant_collection_name
        })
        
        # Create collection if needed
        await qdrant_client.create_collection()
        logger.info("✅ Anime collection ready")
    else:
        logger.error("❌ Qdrant connection failed", extra={
            "url": settings.qdrant_url
        })
    
    yield
    
    # Cleanup (if needed)
    logger.info("🛑 Shutting down MCP server")

# Create FastAPI app with centralized configuration
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# Configure CORS with centralized settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Include API routes
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": settings.api_title,
        "version": settings.api_version,
        "status": "running",
        "endpoints": {
            "search": "/api/search",
            "recommendations": "/api/recommendations",
            "health": "/health",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    qdrant_healthy = await qdrant_client.health_check() if qdrant_client else False
    
    return {
        "status": "healthy" if qdrant_healthy else "unhealthy",
        "qdrant": "connected" if qdrant_healthy else "disconnected",
        "timestamp": datetime.utcnow().isoformat(),
        "configuration": {
            "qdrant_url": settings.qdrant_url,
            "collection_name": settings.qdrant_collection_name,
            "vector_size": settings.qdrant_vector_size
        }
    }

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant client not initialized")
    
    return await qdrant_client.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=settings.host, 
        port=settings.port, 
        reload=settings.debug
    )