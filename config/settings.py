"""Application settings and configuration."""
from pydantic_settings import BaseSettings
from typing import Optional
from urllib.parse import quote_plus


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""
    
    # Database authentication - loaded from .env file
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "123"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "job_recommendation_db"
    
    # Embedding Model - loaded from .env file
    # Using Vietnamese SimCSE model based on PhoBERT (state-of-the-art for Vietnamese)
    EMBEDDING_MODEL: str = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
    EMBEDDING_DIMENSION: int = 768  # PhoBERT base has 768 dimensions
    
    # API - loaded from .env file
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Logging - loaded from .env file
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def get_database_config(self) -> dict:
        """Get database connection components from environment variables."""
        return {
            "username": self.DB_USER,
            "password": self.DB_PASSWORD,
            "host": self.DB_HOST,
            "port": self.DB_PORT,
            "database": self.DB_NAME,
            "url": self.get_database_url()
        }
    
    def get_database_url(self) -> str:
        """Construct DATABASE_URL from individual components."""
        # URL encode password to handle special characters
        encoded_password = quote_plus(self.DB_PASSWORD)
        return f"postgresql://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


settings = Settings()

