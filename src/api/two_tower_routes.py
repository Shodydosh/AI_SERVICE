"""API routes for Two-Tower architecture."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.services.two_tower_matching_service import TwoTowerMatchingService
from src.embeddings.job_tower_encoder import JobTowerEncoder
from src.embeddings.candidate_tower_encoder import CandidateTowerEncoder
from src.database.two_tower_repository import TwoTowerRepository
from src.api.two_tower_schemas import (
    JobSearchRequest,
    JobSearchResponse,
    CandidateSearchRequest,
    CandidateSearchResponse,
    IndexJobRequest,
    IndexCandidateRequest,
    IndexResponse,
    ReindexRequest,
    ReindexResponse,
    HealthResponse,
    JobMatch,
    CandidateMatch,
    FieldScores
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/search/jobs", response_model=JobSearchResponse)
async def search_jobs(
    request: JobSearchRequest,
    db: Session = Depends(get_db)
):
    """Find top matching jobs for a candidate."""
    try:
        matching_service = TwoTowerMatchingService(db, use_faiss=True)
        
        weights = None
        if request.weights:
            weights = {
                'title': request.weights.title,
                'skills': request.weights.skills,
                'experience': request.weights.experience
            }
        
        matches = matching_service.find_jobs_for_candidate(
            candidate_id=request.candidate_id,
            top_k=request.top_k,
            weights=weights
        )
        
        job_matches = [
            JobMatch(
                job_id=match['job_id'],
                title=match.get('title'),
                company=match.get('company'),
                location=match.get('location'),
                score=match['score'],
                explain=FieldScores(**match['explain'])
            )
            for match in matches
        ]
        
        return JobSearchResponse(
            total_matches=len(job_matches),
            matches=job_matches
        )
    except Exception as e:
        logger.error(f"Error searching jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching jobs: {str(e)}"
        )


@router.post("/search/candidates", response_model=CandidateSearchResponse)
async def search_candidates(
    request: CandidateSearchRequest,
    db: Session = Depends(get_db)
):
    """Find top matching candidates for a job."""
    try:
        matching_service = TwoTowerMatchingService(db, use_faiss=True)
        
        weights = None
        if request.weights:
            weights = {
                'title': request.weights.title,
                'skills': request.weights.skills,
                'experience': request.weights.experience
            }
        
        matches = matching_service.find_candidates_for_job(
            job_id=request.job_id,
            top_k=request.top_k,
            weights=weights
        )
        
        candidate_matches = [
            CandidateMatch(
                candidate_id=match['candidate_id'],
                name=match.get('name'),
                email=match.get('email'),
                score=match['score'],
                explain=FieldScores(**match['explain'])
            )
            for match in matches
        ]
        
        return CandidateSearchResponse(
            total_matches=len(candidate_matches),
            matches=candidate_matches
        )
    except Exception as e:
        logger.error(f"Error searching candidates: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching candidates: {str(e)}"
        )


@router.post("/index/job", response_model=IndexResponse)
async def index_job(
    request: IndexJobRequest,
    db: Session = Depends(get_db)
):
    """Index a single job."""
    try:
        job_encoder = JobTowerEncoder()
        embeddings = job_encoder.encode_job(
            title=request.title,
            skills=request.skills,
            requirements=request.requirement
        )
        
        repository = TwoTowerRepository(db)
        repository.create_job(
            job_id=request.job_id,
            title=request.title,
            skills=request.skills,
            requirement=request.requirement,
            company=request.company,
            location=request.location,
            title_embedding=embeddings['title_embedding'],
            skills_embedding=embeddings['skills_embedding'],
            requirement_embedding=embeddings['requirement_embedding']
        )
        
        return IndexResponse(
            status="success",
            job_id=request.job_id,
            message="Job indexed successfully"
        )
    except Exception as e:
        logger.error(f"Error indexing job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error indexing job: {str(e)}"
        )


@router.post("/index/candidate", response_model=IndexResponse)
async def index_candidate(
    request: IndexCandidateRequest,
    db: Session = Depends(get_db)
):
    """Index a single candidate."""
    try:
        candidate_encoder = CandidateTowerEncoder()
        embeddings = candidate_encoder.encode_candidate(
            title=request.title,
            skills=request.skills,
            experience=request.experience
        )
        
        repository = TwoTowerRepository(db)
        repository.create_candidate(
            candidate_id=request.candidate_id,
            title=request.title,
            skills=request.skills,
            experience=request.experience,
            name=request.name,
            email=request.email,
            title_embedding=embeddings['title_embedding'],
            skills_embedding=embeddings['skills_embedding'],
            experience_embedding=embeddings['experience_embedding']
        )
        
        return IndexResponse(
            status="success",
            candidate_id=request.candidate_id,
            message="Candidate indexed successfully"
        )
    except Exception as e:
        logger.error(f"Error indexing candidate: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error indexing candidate: {str(e)}"
        )


@router.post("/reindex", response_model=ReindexResponse)
async def reindex(
    request: ReindexRequest,
    db: Session = Depends(get_db)
):
    """Trigger reindex operation."""
    try:
        repository = TwoTowerRepository(db)
        tracking = repository.create_reindex_tracking(
            reindex_type=request.reindex_type,
            status="pending"
        )
        
        # TODO: Run reindex in background task
        # For now, just return accepted status
        
        return ReindexResponse(
            status="accepted",
            reindex_id=tracking.id,
            message="Reindex job started",
            estimated_time_minutes=30
        )
    except Exception as e:
        logger.error(f"Error starting reindex: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting reindex: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    try:
        from pathlib import Path
        
        repository = TwoTowerRepository(db)
        total_jobs = len(repository.get_all_jobs())
        total_candidates = len(repository.get_all_candidates())
        
        # Check indices
        indices_path = Path("indices/two_tower")
        indices_status = {}
        index_files = [
            "job_title_index.faiss",
            "job_skills_index.faiss",
            "job_requirement_index.faiss",
            "candidate_title_index.faiss",
            "candidate_skills_index.faiss",
            "candidate_experience_index.faiss"
        ]
        
        for index_file in index_files:
            index_name = index_file.replace(".faiss", "")
            if (indices_path / index_file).exists():
                indices_status[index_name] = "loaded"
            else:
                indices_status[index_name] = "not_found"
        
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            indices=indices_status,
            database="connected",
            total_jobs=total_jobs,
            total_candidates=total_candidates
        )
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in health check: {str(e)}"
        )


