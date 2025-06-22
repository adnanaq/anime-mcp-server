# src/main.py - FastAPI MCP Server Entry Point
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import logging

# Import our modules
from .vector.qdrant_client import QdrantClient
from .api import search, recommendations, admin
from .models.anime import DatabaseStats

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Qdrant client
qdrant_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global qdrant_client
    
    # Initialize Qdrant client
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "anime_database")
    
    qdrant_client = QdrantClient(url=qdrant_url, collection_name=collection_name)
    
    # Health check
    if await qdrant_client.health_check():
        logger.info("✅ Qdrant connection established")
        
        # Create collection if needed
        await qdrant_client.create_collection()
        logger.info("✅ Anime collection ready")
    else:
        logger.error("❌ Qdrant connection failed")
    
    yield
    
    # Cleanup (if needed)
    logger.info("🛑 Shutting down MCP server")

# Create FastAPI app
app = FastAPI(
    title="Anime MCP Server",
    description="AI-powered anime search and recommendation system with vector database",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
default_origins = "http://localhost:3000,http://localhost:5173"
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", default_origins)
allowed_origins = allowed_origins_env.split(",")

# Always include localhost for development
localhost_origins = ["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"]
for origin in localhost_origins:
    if origin not in allowed_origins:
        allowed_origins.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["recommendations"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Anime MCP Server",
        "version": "1.0.0",
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
        "timestamp": "2025-01-21"
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
        host=os.getenv("HOST", "0.0.0.0"), 
        port=int(os.getenv("PORT", 8000)), 
        reload=os.getenv("DEBUG", "True").lower() == "true"
    )