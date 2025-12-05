"""Main FastAPI application for Two-Tower Architecture."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from config.settings import settings
from src.api.two_tower_routes import router as two_tower_router
import logging

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Job Recommendation Service - Two-Tower Architecture",
    description="AI-powered job recommendation service using Two-Tower architecture",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Two-Tower routes
app.include_router(two_tower_router, prefix="/api/v2", tags=["two-tower"])

# Mount static files
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Root endpoint."""
    static_file = Path(__file__).parent.parent.parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(str(static_file))
    return {
        "message": "AI Job Recommendation Service - Two-Tower Architecture",
        "version": "2.0.0",
        "docs": "/docs",
        "api": "/api/v2"
    }


@app.on_event("startup")
async def startup_event():
    """Startup event."""
    logger.info("=" * 80)
    logger.info("Two-Tower Architecture API Server Starting...")
    logger.info("=" * 80)
    logger.info(f"API Version: 2.0.0")
    logger.info(f"Host: {settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"Embedding Dimension: {settings.EMBEDDING_DIMENSION}")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event."""
    logger.info("Shutting down Two-Tower API Server...")

