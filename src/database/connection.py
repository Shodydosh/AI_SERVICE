"""Database connection and session management."""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.settings import settings

# Extract database configuration from DATABASE_URL
db_config = settings.get_database_config()

# Create database engine using the full URL
engine = create_engine(
    db_config["url"],
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_database_info() -> dict:
    """Get database connection information (without password)."""
    return {
        "username": db_config["username"],
        "host": db_config["host"],
        "port": db_config["port"],
        "database": db_config["database"]
    }

