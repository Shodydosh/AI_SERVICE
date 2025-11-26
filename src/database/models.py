"""Database models for embeddings."""
from sqlalchemy import Column, Integer, String, Text, ARRAY, Float, DateTime, Index
from sqlalchemy.sql import func
from .connection import Base


class JobDescriptionEmbedding(Base):
    """Model for storing job description embeddings."""
    __tablename__ = "job_description_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    company = Column(String(200))
    description = Column(Text, nullable=False)
    requirements = Column(Text)
    location = Column(String(200))
    embedding = Column(ARRAY(Float), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_job_embedding', 'embedding', postgresql_using='gin'),
    )


class CandidateEmbedding(Base):
    """Model for storing candidate information embeddings."""
    __tablename__ = "candidate_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(200))
    email = Column(String(200))
    skills = Column(Text)
    experience = Column(Text)
    education = Column(Text)
    summary = Column(Text)
    resume_text = Column(Text)
    embedding = Column(ARRAY(Float), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_candidate_embedding', 'embedding', postgresql_using='gin'),
    )


class ProcessedCandidateRecommendation(Base):
    """Model for storing processed candidate recommendations (top 10 jobs per candidate)."""
    __tablename__ = "processed_candidate_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(100), nullable=False, index=True)
    job_id = Column(String(100), nullable=False, index=True)
    similarity_score = Column(Float, nullable=False)
    skills_similarity = Column(Float)  # Field-by-field similarity scores
    experience_similarity = Column(Float)
    desired_job_similarity = Column(Float)
    rank = Column(Integer, nullable=False)  # Rank from 1-10
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_processed_candidate_job', 'candidate_id', 'job_id', unique=True),
        Index('idx_processed_candidate_rank', 'candidate_id', 'rank'),
    )
