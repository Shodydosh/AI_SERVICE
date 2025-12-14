"""Utility for storing explanations in database."""
import json
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.database.models import ProcessedCandidateRecommendation


def save_explanation_to_db(
    db: Session,
    candidate_id: str,
    job_id: str,
    comprehensive_explanation: Dict[str, Any],
    similarity_score: float,
    rank: int,
    skills_similarity: Optional[float] = None,
    experience_similarity: Optional[float] = None,
    desired_job_similarity: Optional[float] = None
) -> Optional[ProcessedCandidateRecommendation]:
    """
    Save comprehensive explanation to database.
    
    Args:
        db: Database session
        candidate_id: Candidate ID
        job_id: Job ID
        comprehensive_explanation: Full explanation dict from ExplanationGenerator
        similarity_score: Combined similarity score
        rank: Recommendation rank (1-10)
        skills_similarity: Skills similarity score (optional)
        experience_similarity: Experience similarity score (optional)
        desired_job_similarity: Desired job similarity score (optional)
    
    Returns:
        ProcessedCandidateRecommendation object or None if error
    """
    try:
        # Extract explanation components
        level1 = comprehensive_explanation.get('levels', {}).get('level1_rule', {})
        level2 = comprehensive_explanation.get('levels', {}).get('level2_embedding', {})
        level3 = comprehensive_explanation.get('levels', {}).get('level3_humanized', {})
        level5 = comprehensive_explanation.get('levels', {}).get('level5_confidence', {})
        
        # Prepare JSON strings
        rule_scores_json = json.dumps(level1, ensure_ascii=False)
        embedding_scores_json = json.dumps(level2.get('embedding_scores', {}), ensure_ascii=False)
        explanation_text = level3.get('explanation_text', '')
        comprehensive_explanation_json = json.dumps(comprehensive_explanation, ensure_ascii=False)
        confidence_score = level5.get('final_confidence', 0.0)
        
        # Check if record exists
        existing = db.query(ProcessedCandidateRecommendation).filter(
            ProcessedCandidateRecommendation.candidate_id == candidate_id,
            ProcessedCandidateRecommendation.job_id == job_id
        ).first()
        
        if existing:
            # Update existing record
            existing.similarity_score = similarity_score
            existing.skills_similarity = skills_similarity
            existing.experience_similarity = experience_similarity
            existing.desired_job_similarity = desired_job_similarity
            existing.rank = rank
            existing.rule_scores = rule_scores_json
            existing.embedding_scores = embedding_scores_json
            existing.explanation_text = explanation_text
            existing.comprehensive_explanation = comprehensive_explanation_json
            existing.confidence_score = confidence_score
            db.commit()
            return existing
        else:
            # Create new record
            new_record = ProcessedCandidateRecommendation(
                candidate_id=candidate_id,
                job_id=job_id,
                similarity_score=similarity_score,
                skills_similarity=skills_similarity,
                experience_similarity=experience_similarity,
                desired_job_similarity=desired_job_similarity,
                rank=rank,
                rule_scores=rule_scores_json,
                embedding_scores=embedding_scores_json,
                explanation_text=explanation_text,
                comprehensive_explanation=comprehensive_explanation_json,
                confidence_score=confidence_score
            )
            db.add(new_record)
            db.commit()
            db.refresh(new_record)
            return new_record
            
    except Exception as e:
        db.rollback()
        print(f"Error saving explanation to DB: {e}")
        return None


def get_explanation_from_db(
    db: Session,
    candidate_id: str,
    job_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve explanation from database.
    
    Args:
        db: Database session
        candidate_id: Candidate ID
        job_id: Job ID
    
    Returns:
        Comprehensive explanation dict or None if not found
    """
    try:
        record = db.query(ProcessedCandidateRecommendation).filter(
            ProcessedCandidateRecommendation.candidate_id == candidate_id,
            ProcessedCandidateRecommendation.job_id == job_id
        ).first()
        
        if record and record.comprehensive_explanation:
            return json.loads(record.comprehensive_explanation)
        return None
    except Exception as e:
        print(f"Error retrieving explanation from DB: {e}")
        return None









