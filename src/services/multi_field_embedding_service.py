"""Service for processing datasets with multi-field embeddings."""
from typing import List, Dict
from sqlalchemy.orm import Session
import pandas as pd
import logging
from tqdm import tqdm
from src.embeddings.multi_field_generator import MultiFieldEmbeddingGenerator
from src.database.multi_field_repository import MultiFieldEmbeddingRepository
from src.data_processing.jd_processor import JDProcessor
from src.data_processing.candidate_processor import CandidateProcessor
from src.utils.three_field_extractor import ThreeFieldExtractor

logger = logging.getLogger(__name__)


class MultiFieldEmbeddingService:
    """Service for multi-field embedding operations."""
    
    def __init__(self, db: Session):
        """
        Initialize multi-field embedding service.
        
        Args:
            db: Database session
        """
        self.db = db
        self.generator = MultiFieldEmbeddingGenerator()
        self.repository = MultiFieldEmbeddingRepository(db)
    
    def process_jd_dataset(
        self,
        file_path: str,
        file_type: str = "csv",
        batch_size: int = 100
    ) -> int:
        """
        Process and store multi-field embeddings for JD dataset.
        
        Args:
            file_path: Path to the dataset file
            file_type: Type of file (csv or json)
            batch_size: Number of records to process and save at a time
        
        Returns:
            Total number of records processed
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
        
        logger.info(f"Processing {total_records} job descriptions with multi-field embeddings...")
        
        total_processed = 0
        total_batches = (total_records + batch_size - 1) // batch_size
        empty_field_stats = {'title': 0, 'skills': 0, 'requirement': 0}
        total_skipped = 0
        
        # Process in batches with progress bar
        pbar = tqdm(range(0, total_records, batch_size), desc="Processing Job Descriptions", unit="batch")
        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, total_records)
            batch_records = records[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            
            pbar.set_postfix({
                'batch': f"{batch_num}/{total_batches}",
                'processed': total_processed,
                'total': total_records
            })
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_records)} records)...")
            
            # Prepare batch data
            batch_data = []
            for record in batch_records:
                def safe_get(key, default=None):
                    value = record.get(key, default)
                    if pd.isna(value) or value == '' or value == 'nan' or str(value).lower() == 'nan':
                        return None
                    if isinstance(value, str):
                        value = value.strip()
                        return None if value == '' else value
                    return value
                
                job_id = str(record.get('job_id', ''))
                
                if not job_id:
                    logger.warning(f"Skipping job: missing job_id")
                    continue
                
                # Convert dict to Series for extractor
                record_series = pd.Series(record)
                
                # Extract exactly 3 fields using ThreeFieldExtractor
                fields = ThreeFieldExtractor.extract_job_fields(record_series)
                
                if not fields['title']:
                    total_skipped += 1
                    empty_field_stats['title'] += 1
                    continue
                
                # Track empty optional fields
                if not fields['skills']:
                    empty_field_stats['skills'] += 1
                if not fields['requirement']:
                    empty_field_stats['requirement'] += 1
                
                batch_data.append({
                    'job_id': job_id,
                    'title': fields['title'],
                    'skills': fields['skills'],
                    'requirement': fields['requirement'],
                    'company': safe_get('company'),
                    'location': safe_get('location')
                })
            
            # Generate embeddings for batch
            logger.info(f"Generating multi-field embeddings for batch {batch_start//batch_size + 1}...")
            
            # Prepare data for batch embedding generation
            jobs_for_embedding = [
                {
                    'title': job['title'],
                    'skills': job['skills'] or '',
                    'requirement': job['requirement'] or ''
                }
                for job in batch_data
            ]
            
            # Generate embeddings
            embeddings = self.generator.generate_job_embeddings_batch(jobs_for_embedding)
            
            # Combine with original data
            batch_embeddings_data = []
            for job, embedding_dict in zip(batch_data, embeddings):
                batch_embeddings_data.append({
                    'job_id': job['job_id'],
                    'title': job['title'],
                    'skills': job['skills'],
                    'requirement': job['requirement'],
                    'title_embedding': embedding_dict['title_embedding'],
                    'skills_embedding': embedding_dict['skills_embedding'],
                    'requirement_embedding': embedding_dict['requirement_embedding'],
                    'company': job['company'],
                    'location': job['location']
                })
            
            # Save to database
            try:
                saved_count = self.repository.create_job_multi_embeddings_batch(
                    batch_embeddings_data,
                    replace_existing=True
                )
                total_processed += saved_count
                pbar.set_postfix({
                    'batch': f"{batch_num}/{total_batches}",
                    'processed': total_processed,
                    'total': total_records
                })
                logger.info(f"✓ Saved batch {batch_num}/{total_batches}: {saved_count} records (Total: {total_processed}/{total_records})")
            except Exception as e:
                logger.error(f"✗ Error saving batch {batch_num}/{total_batches}: {e}")
                pbar.close()
                raise
        
        pbar.close()
        
        # Log summary of empty fields
        if empty_field_stats['skills'] > 0 or empty_field_stats['requirement'] > 0:
            logger.info(f"Empty fields summary: {empty_field_stats['skills']} jobs missing skills, {empty_field_stats['requirement']} jobs missing requirements")
        if total_skipped > 0:
            logger.warning(f"Skipped {total_skipped} jobs due to missing title")
        
        logger.info(f"Successfully processed {total_processed} job descriptions with multi-field embeddings")
        return total_processed
    
    def process_candidate_dataset(
        self,
        file_path: str,
        file_type: str = "csv",
        batch_size: int = 100
    ) -> int:
        """
        Process and store multi-field embeddings for candidate dataset.
        
        Args:
            file_path: Path to the dataset file
            file_type: Type of file (csv or json)
            batch_size: Number of records to process and save at a time
        
        Returns:
            Total number of records processed
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
        
        logger.info(f"Processing {total_records} candidates with multi-field embeddings...")
        
        total_processed = 0
        total_batches = (total_records + batch_size - 1) // batch_size
        empty_field_stats = {'title': 0, 'skills': 0, 'experience': 0}
        total_skipped = 0
        
        # Process in batches with progress bar
        pbar = tqdm(range(0, total_records, batch_size), desc="Processing Candidates", unit="batch")
        for batch_start in pbar:
            batch_end = min(batch_start + batch_size, total_records)
            batch_records = records[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            
            pbar.set_postfix({
                'batch': f"{batch_num}/{total_batches}",
                'processed': total_processed,
                'total': total_records
            })
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_records)} records)...")
            
            # Prepare batch data
            batch_data = []
            for record in batch_records:
                def safe_get(key, default=None):
                    value = record.get(key, default)
                    if pd.isna(value) or value == '' or value == 'nan' or str(value).lower() == 'nan':
                        return None
                    if isinstance(value, str):
                        value = value.strip()
                        return None if value == '' else value
                    return value
                
                candidate_id = str(record.get('candidate_id', '')) or str(record.get('cv_id', ''))
                
                if not candidate_id:
                    logger.warning(f"Skipping candidate: missing candidate_id")
                    continue
                
                # Convert dict to Series for extractor
                record_series = pd.Series(record)
                
                # Extract exactly 3 fields using ThreeFieldExtractor
                fields = ThreeFieldExtractor.extract_candidate_fields(record_series)
                
                # Track empty fields
                if not fields['title']:
                    empty_field_stats['title'] += 1
                if not fields['skills']:
                    empty_field_stats['skills'] += 1
                if not fields['experience']:
                    empty_field_stats['experience'] += 1
                
                batch_data.append({
                    'candidate_id': candidate_id,
                    'title': fields['title'],
                    'skills': fields['skills'],
                    'experience': fields['experience'],
                    'name': safe_get('name') or safe_get('user_name'),
                    'email': safe_get('email')
                })
            
            # Generate embeddings for batch
            logger.info(f"Generating multi-field embeddings for batch {batch_start//batch_size + 1}...")
            
            # Prepare data for batch embedding generation
            candidates_for_embedding = [
                {
                    'title': candidate['title'] or '',
                    'skills': candidate['skills'] or '',
                    'experience': candidate['experience'] or ''
                }
                for candidate in batch_data
            ]
            
            # Generate embeddings
            embeddings = self.generator.generate_candidate_embeddings_batch(candidates_for_embedding)
            
            # Combine with original data
            batch_embeddings_data = []
            for candidate, embedding_dict in zip(batch_data, embeddings):
                batch_embeddings_data.append({
                    'candidate_id': candidate['candidate_id'],
                    'title': candidate['title'],
                    'skills': candidate['skills'],
                    'experience': candidate['experience'],
                    'title_embedding': embedding_dict['title_embedding'],
                    'skills_embedding': embedding_dict['skills_embedding'],
                    'experience_embedding': embedding_dict['experience_embedding'],
                    'name': candidate['name'],
                    'email': candidate['email']
                })
            
            # Save to database
            try:
                saved_count = self.repository.create_candidate_multi_embeddings_batch(
                    batch_embeddings_data,
                    replace_existing=True
                )
                total_processed += saved_count
                pbar.set_postfix({
                    'batch': f"{batch_num}/{total_batches}",
                    'processed': total_processed,
                    'total': total_records
                })
                logger.info(f"✓ Saved batch {batch_num}/{total_batches}: {saved_count} records (Total: {total_processed}/{total_records})")
            except Exception as e:
                logger.error(f"✗ Error saving batch {batch_num}/{total_batches}: {e}")
                pbar.close()
                raise
        
        pbar.close()
        
        # Log summary of empty fields
        if empty_field_stats['title'] > 0 or empty_field_stats['skills'] > 0 or empty_field_stats['experience'] > 0:
            logger.info(f"Empty fields summary: {empty_field_stats['title']} candidates missing title, {empty_field_stats['skills']} missing skills, {empty_field_stats['experience']} missing experience")
        if total_skipped > 0:
            logger.warning(f"Skipped {total_skipped} candidates due to missing candidate_id")
        
        logger.info(f"Successfully processed {total_processed} candidates with multi-field embeddings")
        return total_processed
