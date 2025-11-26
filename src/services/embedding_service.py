"""Service for managing embeddings."""
from typing import List, Dict
from sqlalchemy.orm import Session
import pandas as pd
from src.embeddings.generator import EmbeddingGenerator
from src.embeddings.weighted_embedding import WeightedEmbeddingGenerator
from src.data_processing.jd_processor import JDProcessor
from src.data_processing.candidate_processor import CandidateProcessor
from src.database.repository import EmbeddingRepository
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for embedding operations."""
    
    def __init__(self, db: Session, use_weighted: bool = True):
        """
        Initialize embedding service.
        
        Args:
            db: Database session
            use_weighted: Whether to use weighted embeddings (default: True)
        """
        self.db = db
        self.use_weighted = use_weighted
        if use_weighted:
            self.embedding_generator = WeightedEmbeddingGenerator()
            logger.info("Using weighted embedding generator")
        else:
            self.embedding_generator = EmbeddingGenerator()
        self.repository = EmbeddingRepository(db)
    
    def process_jd_dataset(self, file_path: str, file_type: str = "csv", batch_size: int = 1000) -> int:
        """Process and store embeddings for JD dataset in batches.
        
        Args:
            file_path: Path to the dataset file
            file_type: Type of file (csv or json)
            batch_size: Number of records to process and save at a time (default: 1000)
        
        Returns:
            Total number of records processed
        
        Raises:
            Exception: If database error occurs, processing stops immediately
        """
        processor = JDProcessor()
        
        if file_type.lower() == "csv":
            processor.load_from_csv(file_path)
        elif file_type.lower() == "json":
            processor.load_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        if not processor.validate_data():
            raise ValueError("JD dataset validation failed")
        
        records = processor.get_records()
        total_records = len(records)
        
        logger.info(f"Processing {total_records} job descriptions in batches of {batch_size}...")
        
        total_processed = 0
        
        # Process in batches
        for batch_start in range(0, total_records, batch_size):
            batch_end = min(batch_start + batch_size, total_records)
            batch_records = records[batch_start:batch_end]
            
            # Generate embeddings for this batch
            logger.info(f"Generating embeddings for batch {batch_start//batch_size + 1} ({len(batch_records)} records)...")
            
            if self.use_weighted and isinstance(self.embedding_generator, WeightedEmbeddingGenerator):
                # Use weighted embeddings for JD
                batch_embeddings = []
                for record in batch_records:
                    field_texts = processor.get_field_texts(pd.Series(record))
                    weights = WeightedEmbeddingGenerator.DEFAULT_JD_WEIGHTS
                    embedding = self.embedding_generator.generate_weighted_embedding(
                        field_texts=field_texts,
                        weights=weights,
                        method="repetition"
                    )
                    batch_embeddings.append(embedding)
            else:
                # Use standard embeddings
                batch_texts = [processor.get_combined_text(pd.Series(record)) for record in batch_records]
                batch_embeddings = self.embedding_generator.generate_embeddings_batch(batch_texts)
            
            # Prepare batch data for database
            batch_data = []
            for record, embedding in zip(batch_records, batch_embeddings):
                # Helper function to safely extract and convert NaN to None
                def safe_get(key, default=None):
                    value = record.get(key, default)
                    # Convert pandas NaN, numpy NaN, or empty strings to None
                    if pd.isna(value) or value == '' or value == 'nan' or str(value).lower() == 'nan':
                        return None
                    # Convert to string and strip whitespace, return None if empty
                    if isinstance(value, str):
                        value = value.strip()
                        return None if value == '' else value
                    return value
                
                batch_data.append({
                    'job_id': str(record.get('job_id', '')),
                    'title': safe_get('title') or '',
                    'description': safe_get('description') or '',
                    'embedding': embedding,
                    'company': safe_get('company'),
                    'requirements': safe_get('requirements'),
                    'location': safe_get('location')
                })
            
            # Save batch to database - stop on error
            try:
                saved_count = self.repository.create_jd_embeddings_batch(batch_data, replace_existing=True)
                total_processed += saved_count
                logger.info(f"✓ Saved batch {batch_start//batch_size + 1}: {saved_count} records (Total: {total_processed}/{total_records})")
            except Exception as e:
                logger.error(f"✗ Database error saving batch {batch_start//batch_size + 1}: {e}")
                logger.error(f"Stopping processing. Successfully processed {total_processed} records before error.")
                raise  # Stop immediately on database error
        
        logger.info(f"Successfully processed {total_processed} job descriptions")
        return total_processed
    
    def process_candidate_dataset(self, file_path: str, file_type: str = "csv", batch_size: int = 1000) -> int:
        """Process and store embeddings for candidate dataset in batches.
        
        Args:
            file_path: Path to the dataset file
            file_type: Type of file (csv or json)
            batch_size: Number of records to process and save at a time (default: 1000)
        
        Returns:
            Total number of records processed
        
        Raises:
            Exception: If database error occurs, processing stops immediately
        """
        processor = CandidateProcessor()
        
        if file_type.lower() == "csv":
            processor.load_from_csv(file_path)
        elif file_type.lower() == "json":
            processor.load_from_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        if not processor.validate_data():
            raise ValueError("Candidate dataset validation failed")
        
        records = processor.get_records()
        total_records = len(records)
        
        logger.info(f"Processing {total_records} candidates in batches of {batch_size}...")
        
        total_processed = 0
        
        # Process in batches
        for batch_start in range(0, total_records, batch_size):
            batch_end = min(batch_start + batch_size, total_records)
            batch_records = records[batch_start:batch_end]
            
            # Generate embeddings for this batch
            logger.info(f"Generating embeddings for batch {batch_start//batch_size + 1} ({len(batch_records)} records)...")
            
            if self.use_weighted and isinstance(self.embedding_generator, WeightedEmbeddingGenerator):
                # Use weighted embeddings for candidates with dynamic weights
                batch_embeddings = []
                for record in batch_records:
                    field_texts = processor.get_field_texts(pd.Series(record))
                    # Use dynamic weights to adjust based on available fields
                    embedding = self.embedding_generator.generate_weighted_embedding(
                        field_texts=field_texts,
                        weights=None,  # Will use default and apply dynamic adjustment
                        method="repetition",
                        use_dynamic_weights=True  # Enable dynamic weight adjustment
                    )
                    batch_embeddings.append(embedding)
            else:
                # Use standard embeddings
                batch_texts = [processor.get_combined_text(pd.Series(record)) for record in batch_records]
                batch_embeddings = self.embedding_generator.generate_embeddings_batch(batch_texts)
            
            # Prepare batch data for database
            batch_data = []
            for record, embedding in zip(batch_records, batch_embeddings):
                # Helper function to safely extract and convert NaN to None
                def safe_get(key, default=None):
                    value = record.get(key, default)
                    
                    # Handle None
                    if value is None:
                        return None
                    
                    # Check for pandas/numpy NaN
                    try:
                        if pd.isna(value):
                            return None
                    except (TypeError, ValueError):
                        pass
                    
                    # Convert to string and check for NaN strings (case-insensitive)
                    str_value = str(value).strip()
                    str_lower = str_value.lower()
                    if str_value == '' or str_lower == 'nan' or str_lower == 'none' or str_lower == 'null':
                        return None
                    
                    # Return cleaned string or original value
                    return str_value if isinstance(value, str) else value
                
                candidate_id = str(record.get('candidate_id', ''))
                name = safe_get('name')
                email = safe_get('email')
                skills = safe_get('skills')
                
                # Log first record of each batch for verification
                if len(batch_data) == 0:
                    logger.info(f"Sample record - ID: {candidate_id}, Name: {name}, Email: {email}, Skills: {skills[:50] if skills else 'None'}...")
                
                batch_data.append({
                    'candidate_id': candidate_id,
                    'embedding': embedding,
                    'name': name,
                    'email': email,
                    'skills': skills,
                    'experience': safe_get('experience'),
                    'education': safe_get('education'),
                    'summary': safe_get('summary'),
                    'resume_text': safe_get('resume_text')
                })
            
            # Save batch to database - stop on error
            try:
                saved_count = self.repository.create_candidate_embeddings_batch(batch_data, replace_existing=True)
                total_processed += saved_count
                logger.info(f"✓ Saved batch {batch_start//batch_size + 1}: {saved_count} records (Total: {total_processed}/{total_records})")
            except Exception as e:
                logger.error(f"✗ Database error saving batch {batch_start//batch_size + 1}: {e}")
                logger.error(f"Stopping processing. Successfully processed {total_processed} records before error.")
                raise  # Stop immediately on database error
        
        logger.info(f"Successfully processed {total_processed} candidates")
        return total_processed
    
    def recommend_jobs_for_candidate(self, candidate_id: str, limit: int = 10) -> List[Dict]:
        """Recommend jobs for a candidate."""
        jobs = self.repository.recommend_jobs_for_candidate(candidate_id, limit)
        return [
            {
                "job_id": job.job_id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "description": job.description[:500] if job.description else None
            }
            for job in jobs
        ]
    
    def recommend_candidates_for_job(self, job_id: str, limit: int = 10) -> List[Dict]:
        """Recommend candidates for a job."""
        candidates = self.repository.recommend_candidates_for_job(job_id, limit)
        return [
            {
                "candidate_id": candidate.candidate_id,
                "name": candidate.name,
                "email": candidate.email,
                "skills": candidate.skills,
                "summary": candidate.summary[:500] if candidate.summary else None
            }
            for candidate in candidates
        ]

