"""Database models for embedding evaluation research."""
from sqlalchemy import Column, Integer, String, Text, ARRAY, Float, DateTime, Index
from sqlalchemy.sql import func
from .connection import Base


class EmbeddingEvaluationJD(Base):
    """Model for storing JD embeddings from different methods."""
    __tablename__ = "embedding_evaluation_jd"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), nullable=False, index=True)
    method_id = Column(Integer, nullable=False, index=True)  # 1-5
    method_name = Column(String(50), nullable=False)
    
    # Original data
    title = Column(String(500))
    company = Column(String(200))
    description = Column(Text)
    requirements = Column(Text)
    location = Column(String(200))
    
    # Embeddings from different methods
    embedding = Column(ARRAY(Float), nullable=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_eval_jd_method', 'job_id', 'method_id', unique=True),
        Index('idx_eval_jd_method_name', 'method_name'),
    )


class EmbeddingEvaluationCandidate(Base):
    """Model for storing candidate embeddings from different methods."""
    __tablename__ = "embedding_evaluation_candidate"
    
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(100), nullable=False, index=True)
    method_id = Column(Integer, nullable=False, index=True)  # 1-5
    method_name = Column(String(50), nullable=False)
    
    # Original data
    name = Column(String(200))
    email = Column(String(200))
    skills = Column(Text)
    experience = Column(Text)
    education = Column(Text)
    summary = Column(Text)
    
    # Embeddings from different methods
    embedding = Column(ARRAY(Float), nullable=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_eval_candidate_method', 'candidate_id', 'method_id', unique=True),
        Index('idx_eval_candidate_method_name', 'method_name'),
    )


class EmbeddingEvaluationResults(Base):
    """Model for storing evaluation results."""
    __tablename__ = "embedding_evaluation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    method_id = Column(Integer, nullable=False, index=True)
    method_name = Column(String(50), nullable=False)
    
    # Evaluation metrics
    top_1_accuracy = Column(Float)
    top_5_accuracy = Column(Float)
    top_10_accuracy = Column(Float)
    mrr = Column(Float)  # Mean Reciprocal Rank
    ndcg_5 = Column(Float)
    ndcg_10 = Column(Float)
    precision_5 = Column(Float)
    precision_10 = Column(Float)
    
    # Performance metrics
    avg_generation_time_ms = Column(Float)
    avg_search_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    storage_size_mb = Column(Float)
    
    # Quality metrics
    avg_similarity_score = Column(Float)
    similarity_std = Column(Float)
    high_similarity_coverage = Column(Float)  # % with >0.8 similarity
    
    # Test details
    test_samples = Column(Integer)
    evaluation_date = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_eval_results_method', 'method_id', unique=True),
    )

