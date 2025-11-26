"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from src.api.routes import router
from config.settings import settings
from src.services.scheduler_service import SchedulerService
import logging
import atexit

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Job Recommendation Service",
    description="AI-powered job recommendation service using embeddings",
    version="1.0.0"
)

# Initialize scheduler (exported for use in routes)
scheduler_service = SchedulerService()

# Export scheduler for use in other modules
def get_scheduler_service() -> SchedulerService:
    """Get the scheduler service instance."""
    return scheduler_service

# Start scheduler on app startup
@app.on_event("startup")
async def startup_event():
    """Start scheduler on application startup."""
    logger.info("Starting scheduler service...")
    scheduler_service.start()
    
    # Add regeneration job to run every 12 hours
    # This will regenerate embeddings and recompute recommendations
    scheduler_service.add_regeneration_job(
        hours=12,
        jd_file=None,  # Will use existing data in database
        candidate_file=None
    )
    
    logger.info("Scheduler started with regeneration job (every 12 hours)")

# Stop scheduler on app shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Stop scheduler on application shutdown."""
    logger.info("Stopping scheduler service...")
    scheduler_service.stop()
    logger.info("Scheduler stopped")

# Register cleanup on exit
atexit.register(lambda: scheduler_service.stop())

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["recommendations"])

# Mount static files
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Root endpoint - serve UI."""
    static_file = Path(__file__).parent.parent.parent / "static" / "index.html"
    if static_file.exists():
        return FileResponse(str(static_file))
    return {
        "message": "AI Job Recommendation Service",
        "version": "1.0.0",
        "docs": "/docs",
        "ui": "/static/index.html"
    }

