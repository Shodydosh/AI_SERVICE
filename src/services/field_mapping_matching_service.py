"""Matching service using field-by-field embedding mapping with multi-filtering."""
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
import logging
import numpy as np
from src.embeddings.field_mapping_embedding import FieldMappingEmbeddingGenerator
from src.database.repository import EmbeddingRepository
from src.data_processing.candidate_processor import CandidateProcessor
from src.data_processing.jd_processor import JDProcessor
import pandas as pd

logger = logging.getLogger(__name__)


class FieldMappingMatchingService:
    """Service for candidate-to-job matching using field-by-field embeddings with multi-filtering."""
    
    def __init__(self, db: Session, model_name: str = None):
        """
        Initialize field mapping matching service.
        
        Args:
            db: Database session
            model_name: Optional embedding model name
        """
        self.db = db
        self.repository = EmbeddingRepository(db)
        self.embedding_generator = FieldMappingEmbeddingGenerator(model_name=model_name)
        
        # Field weights for combining similarities
        self.field_weights = {
            'skills': 0.4,        # Skills are most important
            'experience': 0.35,    # Experience is very important
            'desired_job': 0.25    # Desired job is important but less than skills/experience
        }
    
    def extract_candidate_fields(self, candidate_row: pd.Series) -> Dict[str, str]:
        """
        Extract relevant fields from candidate row with improved fallback logic.
        
        Args:
            candidate_row: Pandas Series with candidate data
        
        Returns:
            Dictionary of field_name -> text content
        """
        fields = {}
        
        def get_field(field_name: str, alternatives: List[str] = None) -> Optional[str]:
            """Get field value, trying alternatives if main field not found."""
            # Try main field first
            if field_name in candidate_row.index:
                try:
                    value = candidate_row[field_name]
                    if not pd.isna(value) and value is not None:
                        text = str(value).strip()
                        if text:
                            return text
                except (KeyError, AttributeError):
                    pass
            
            # Try alternatives
            if alternatives:
                for alt_field in alternatives:
                    if alt_field in candidate_row.index:
                        try:
                            value = candidate_row[alt_field]
                            if not pd.isna(value) and value is not None:
                                text = str(value).strip()
                                if text:
                                    return text
                        except (KeyError, AttributeError):
                            continue
            
            return None
        
        # Extract skills with multiple alternatives
        skills = get_field('skills', ['Skills', 'skill', 'technical_skills', 'competencies', 'summary'])
        if skills:
            fields['skills'] = skills
        
        # Extract experience with alternatives
        experience = get_field('experience', ['work_experience', 'Experience', 'work_history', 'employment_history', 'professional_experience'])
        if experience:
            fields['experience'] = experience
        
        # Extract desired job with alternatives
        desired_job = get_field('desired_job_translated', [
            'desired_job', 'Desired Job', 'target_position', 'preferred_job', 
            'job_preference', 'career_goal', 'objective'
        ])
        if desired_job:
            fields['desired_job'] = desired_job
        
        # Fallback: if no fields found, try to use summary or resume_text
        if not fields:
            summary = get_field('summary', ['Summary', 'resume_text', 'Resume Text', 'profile'])
            if summary:
                # Use summary as experience fallback
                fields['experience'] = summary[:500]  # Limit length
        
        return fields
    
    def extract_jd_fields(self, jd_row: pd.Series) -> Dict[str, str]:
        """
        Extract relevant fields from JD row.
        
        Args:
            jd_row: Pandas Series with JD data
        
        Returns:
            Dictionary of field_name -> text content
        """
        fields = {}
        
        def get_field(field_name: str) -> Optional[str]:
            if field_name not in jd_row.index:
                return None
            try:
                value = jd_row[field_name]
                if pd.isna(value) or value is None:
                    return None
                text = str(value).strip()
                return text if text else None
            except (KeyError, AttributeError):
                return None
        
        # Extract mapped fields
        requirements = get_field('requirements')
        if requirements:
            fields['requirements'] = requirements
        
        title = get_field('title')
        if title:
            fields['title'] = title
        
        return fields
    
    def find_top_jobs_for_candidate(
        self,
        candidate_id: str,
        top_k: int = 10,
        candidate_file: Optional[str] = None,
        jd_file: Optional[str] = None
    ) -> List[Dict]:
        """
        Find top K jobs for a candidate using multi-filtering approach.
        
        Args:
            candidate_id: Candidate ID
            top_k: Number of top jobs to return (default: 10)
            candidate_file: Optional path to candidate CSV file
            jd_file: Optional path to JD CSV file
        
        Returns:
            List of top K job matches with similarity scores and field breakdowns
        """
        # Load candidate data
        if candidate_file:
            candidate_processor = CandidateProcessor()
            candidate_processor.load_from_csv(candidate_file)
            candidate_data = candidate_processor.data
        else:
            # Try to get from database
            candidate_embedding = self.repository.get_candidate_embedding(candidate_id)
            if not candidate_embedding:
                logger.error(f"Candidate {candidate_id} not found")
                return []
            
            # For database candidates, we need to reconstruct fields
            # This is a limitation - we should store field embeddings separately
            logger.warning("Candidate from database - field extraction may be limited")
            return []
        
        # Find candidate row
        candidate_rows = candidate_data[candidate_data['candidate_id'] == candidate_id]
        if candidate_rows.empty:
            logger.error(f"Candidate {candidate_id} not found in data")
            return []
        
        candidate_row = candidate_rows.iloc[0]
        candidate_fields = self.extract_candidate_fields(candidate_row)
        
        if not candidate_fields:
            logger.error(f"No relevant fields found for candidate {candidate_id}")
            return []
        
        logger.info(f"Found candidate fields: {list(candidate_fields.keys())}")
        
        # Generate candidate field embeddings
        candidate_embeddings = self.embedding_generator.generate_candidate_field_embeddings(candidate_fields)
        
        if not candidate_embeddings:
            logger.error(f"Could not generate embeddings for candidate {candidate_id}")
            return []
        
        logger.info(f"Generated embeddings for {len(candidate_embeddings)} candidate fields")
        
        # Load JD data
        if jd_file:
            jd_processor = JDProcessor()
            jd_processor.load_from_csv(jd_file)
            jd_data = jd_processor.data
        else:
            # Get from database
            all_jds = self.repository.get_all_jd_embeddings()
            if not all_jds:
                logger.error("No JDs found in database")
                return []
            
            # Convert to DataFrame for processing
            jd_records = []
            for jd in all_jds:
                jd_records.append({
                    'job_id': jd.job_id,
                    'title': jd.title,
                    'requirements': jd.requirements,
                    'description': jd.description,
                    'company': jd.company,
                    'location': jd.location
                })
            jd_data = pd.DataFrame(jd_records)
        
        logger.info(f"Processing {len(jd_data)} job descriptions")
        
        # Multi-filtering: Compare each candidate field with corresponding JD fields
        job_scores = []
        
        for idx, jd_row in jd_data.iterrows():
            jd_fields = self.extract_jd_fields(jd_row)
            
            if not jd_fields:
                continue
            
            # Generate JD field embeddings
            jd_embeddings = self.embedding_generator.generate_jd_field_embeddings(jd_fields)
            
            if not jd_embeddings:
                continue
            
            # Calculate field-by-field similarities
            combined_similarity, field_similarities = self.embedding_generator.calculate_combined_similarity(
                candidate_embeddings,
                jd_embeddings,
                weights=self.field_weights
            )
            
            # Store result with field breakdown
            job_scores.append({
                'job_id': jd_row.get('job_id', f'jd_{idx}'),
                'title': jd_row.get('title', ''),
                'company': jd_row.get('company', ''),
                'location': jd_row.get('location', ''),
                'description': jd_row.get('description', '')[:500] if jd_row.get('description') else '',
                'requirements': jd_row.get('requirements', '')[:300] if jd_row.get('requirements') else '',
                'similarity_score': round(combined_similarity, 4),
                'field_similarities': field_similarities
            })
        
        # Sort by combined similarity score (descending)
        job_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top K
        top_jobs = job_scores[:top_k]
        
        logger.info(f"Found top {len(top_jobs)} jobs for candidate {candidate_id}")
        if top_jobs:
            logger.info(f"Top similarity: {top_jobs[0]['similarity_score']:.4f}")
            logger.info(f"Field breakdown: {top_jobs[0]['field_similarities']}")
        
        return top_jobs
    
    def find_top_jobs_for_all_candidates(
        self,
        candidate_file: str,
        jd_file: str,
        top_k: int = 10,
        limit_candidates: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """
        Find top K jobs for all candidates using multi-filtering.
        
        Args:
            candidate_file: Path to candidate CSV file
            jd_file: Path to JD CSV file
            top_k: Number of top jobs per candidate (default: 10)
            limit_candidates: Optional limit on number of candidates to process
        
        Returns:
            Dictionary of candidate_id -> list of top K job matches
        """
        # Load data
        candidate_processor = CandidateProcessor()
        candidate_processor.load_from_csv(candidate_file)
        candidate_data = candidate_processor.data
        
        jd_processor = JDProcessor()
        jd_processor.load_from_csv(jd_file)
        jd_data = jd_processor.data
        
        # Limit candidates if specified
        if limit_candidates:
            candidate_data = candidate_data.head(limit_candidates)
        
        logger.info(f"Processing {len(candidate_data)} candidates against {len(jd_data)} jobs")
        
        # Pre-generate JD field embeddings for efficiency (improved batch processing)
        logger.info("Pre-generating JD field embeddings...")
        jd_embeddings_cache = {}
        
        # Collect all JD fields first
        jd_fields_list = []
        jd_rows_list = []
        jd_ids_list = []
        
        for idx, jd_row in jd_data.iterrows():
            jd_fields = self.extract_jd_fields(jd_row)
            if jd_fields:
                jd_id = jd_row.get('job_id', f'jd_{idx}')
                jd_fields_list.append(jd_fields)
                jd_rows_list.append(jd_row)
                jd_ids_list.append(jd_id)
        
        # Generate embeddings in batch for better performance
        if jd_fields_list:
            logger.info(f"Generating embeddings for {len(jd_fields_list)} jobs...")
            for i, (jd_fields, jd_row, jd_id) in enumerate(zip(jd_fields_list, jd_rows_list, jd_ids_list)):
                try:
                    jd_embeddings = self.embedding_generator.generate_jd_field_embeddings(jd_fields)
                    if jd_embeddings:
                        jd_embeddings_cache[jd_id] = {
                            'embeddings': jd_embeddings,
                            'fields': jd_fields,
                            'row': jd_row
                        }
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"  Processed {i + 1}/{len(jd_fields_list)} jobs")
                except Exception as e:
                    logger.warning(f"Error generating embeddings for JD {jd_id}: {e}")
                    continue
        
        logger.info(f"Cached embeddings for {len(jd_embeddings_cache)} jobs")
        
        # Process each candidate
        results = {}
        
        for idx, candidate_row in candidate_data.iterrows():
            candidate_id = candidate_row.get('candidate_id', f'candidate_{idx}')
            
            try:
                candidate_fields = self.extract_candidate_fields(candidate_row)
                
                if not candidate_fields:
                    logger.warning(f"No fields found for candidate {candidate_id}")
                    results[candidate_id] = []
                    continue
                
                # Generate candidate field embeddings
                candidate_embeddings = self.embedding_generator.generate_candidate_field_embeddings(candidate_fields)
                
                if not candidate_embeddings:
                    logger.warning(f"Could not generate embeddings for candidate {candidate_id}")
                    results[candidate_id] = []
                    continue
                
                # Compare with all JDs
                job_scores = []
                
                for jd_id, jd_data in jd_embeddings_cache.items():
                    jd_embeddings = jd_data['embeddings']
                    jd_row = jd_data['row']
                    
                    if not jd_embeddings:
                        continue
                    
                    # Calculate similarities
                    combined_similarity, field_similarities = self.embedding_generator.calculate_combined_similarity(
                        candidate_embeddings,
                        jd_embeddings,
                        weights=self.field_weights
                    )
                    
                    job_scores.append({
                        'job_id': jd_id,
                        'title': jd_row.get('title', ''),
                        'company': jd_row.get('company', ''),
                        'location': jd_row.get('location', ''),
                        'description': jd_row.get('description', '')[:500] if jd_row.get('description') else '',
                        'requirements': jd_row.get('requirements', '')[:300] if jd_row.get('requirements') else '',
                        'similarity_score': round(combined_similarity, 4),
                        'field_similarities': field_similarities
                    })
                
                # Sort and get top K
                job_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
                top_jobs = job_scores[:top_k]
                
                results[candidate_id] = top_jobs
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"Processed {idx + 1}/{len(candidate_data)} candidates")
            
            except Exception as e:
                logger.error(f"Error processing candidate {candidate_id}: {e}")
                results[candidate_id] = []
        
        logger.info(f"Completed processing {len(results)} candidates")
        return results

