"""API routes for the job recommendation service."""
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.services.embedding_service import EmbeddingService
from src.api import schemas
from src.api.schemas import (
    JobRecommendationRequest,
    CandidateRecommendationRequest,
    JobRecommendationResponse,
    CandidateRecommendationResponse,
    DatasetProcessRequest,
    DatasetProcessResponse,
    JobInfo,
    CandidateInfo,
    CandidateTextRequest,
    CandidateInputRequest,
    CandidateIdRequest,
    CandidateFileRequest,
    JobMatchResponse,
    JobMatchInfo,
    CandidateListItem,
    CandidateListResponse,
    JobIdsResponse,
    MultiFilterJobMatchInfo,
    MultiFilterJobMatchResponse,
    MultiFilterCandidateRequest,
    CandidateCreateRequest,
    CandidateCreateResponse,
    APIResponse
)
from src.api.background_tasks import update_faiss_and_recommend, update_faiss_for_batch
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process/jd-dataset", response_model=DatasetProcessResponse)
async def process_jd_dataset(
    request: DatasetProcessRequest,
    db: Session = Depends(get_db)
):
    """Process and store embeddings for JD dataset."""
    try:
        service = EmbeddingService(db)
        count = service.process_jd_dataset(request.file_path, request.file_type)
        return DatasetProcessResponse(
            message="JD dataset processed successfully",
            records_processed=count
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing JD dataset: {str(e)}"
        )


@router.post("/process/candidate-dataset", response_model=DatasetProcessResponse)
async def process_candidate_dataset(
    request: DatasetProcessRequest,
    db: Session = Depends(get_db)
):
    """Process and store embeddings for candidate dataset."""
    try:
        service = EmbeddingService(db)
        count = service.process_candidate_dataset(request.file_path, request.file_type)
        return DatasetProcessResponse(
            message="Candidate dataset processed successfully",
            records_processed=count
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing candidate dataset: {str(e)}"
        )


@router.post("/recommend/jobs", response_model=JobRecommendationResponse)
async def recommend_jobs(
    request: JobRecommendationRequest,
    db: Session = Depends(get_db)
):
    """Get job recommendations for a candidate."""
    try:
        from src.services.matching_service import MatchingService
        
        matching_service = MatchingService(db, use_faiss=True, use_reranking=True)
        recommendations = matching_service.find_jobs_for_candidate(
            request.candidate_id,
            top_k=request.limit
        )
        
        job_infos = [
            JobInfo(
                job_id=job["job_id"],
                title=job["title"],
                company=job.get("company"),
                location=job.get("location"),
                description=job.get("description")
            )
            for job in recommendations
        ]
        
        return JobRecommendationResponse(
            candidate_id=request.candidate_id,
            recommendations=job_infos
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting job recommendations: {str(e)}"
        )


@router.post("/recommend/candidates", response_model=CandidateRecommendationResponse)
async def recommend_candidates(
    request: CandidateRecommendationRequest,
    db: Session = Depends(get_db)
):
    """Get candidate recommendations for a job."""
    try:
        service = EmbeddingService(db)
        recommendations = service.recommend_candidates_for_job(
            request.job_id,
            request.limit
        )
        
        candidate_infos = [
            CandidateInfo(
                candidate_id=candidate["candidate_id"],
                name=candidate.get("name"),
                email=candidate.get("email"),
                skills=candidate.get("skills"),
                summary=candidate.get("summary")
            )
            for candidate in recommendations
        ]
        
        return CandidateRecommendationResponse(
            job_id=request.job_id,
            recommendations=candidate_infos
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting candidate recommendations: {str(e)}"
        )


@router.post("/match/candidate-text", response_model=JobMatchResponse)
async def match_candidate_text(
    request: CandidateTextRequest,
    db: Session = Depends(get_db)
):
    """Find top matching jobs for candidate text."""
    try:
        from src.services.matching_service import MatchingService
        
        matching_service = MatchingService(db, use_faiss=True, use_reranking=True)
        matches = matching_service.find_jobs_for_candidate_text(
            request.candidate_text,
            top_k=request.limit
        )
        
        job_matches = [
            JobMatchInfo(
                job_id=match["job_id"],
                title=match["title"],
                company=match.get("company"),
                location=match.get("location"),
                description=match.get("description"),
                requirements=match.get("requirements"),
                similarity_score=match["similarity_score"]
            )
            for match in matches
        ]
        
        return JobMatchResponse(
            total_matches=len(job_matches),
            matches=job_matches
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error matching candidate: {str(e)}"
        )


@router.post("/match/candidate-file", response_model=JobMatchResponse)
async def match_candidate_from_file(
    request: CandidateFileRequest,
    db: Session = Depends(get_db)
):
    """Find top matching jobs for candidate from processed file."""
    try:
        from src.services.matching_service import MatchingService
        
        matching_service = MatchingService(db, use_faiss=True, use_reranking=True)
        matches = matching_service.find_jobs_for_candidate_from_file(
            request.candidate_file,
            candidate_index=request.candidate_index,
            top_k=request.limit
        )
        
        job_matches = [
            JobMatchInfo(
                job_id=match["job_id"],
                title=match["title"],
                company=match.get("company"),
                location=match.get("location"),
                description=match.get("description"),
                requirements=match.get("requirements"),
                similarity_score=match["similarity_score"]
            )
            for match in matches
        ]
        
        return JobMatchResponse(
            candidate_index=request.candidate_index,
            total_matches=len(job_matches),
            matches=job_matches
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error matching candidate from file: {str(e)}"
        )


@router.get("/candidates", response_model=CandidateListResponse)
async def get_candidates(
    db: Session = Depends(get_db),
    limit: int = 1000
):
    """Get list of processed candidates from database."""
    try:
        from src.database.repository import EmbeddingRepository
        
        repository = EmbeddingRepository(db)
        candidates = repository.get_all_candidate_embeddings()
        
        # Limit results
        candidates = candidates[:limit]
        
        candidate_list = []
        for candidate in candidates:
            # Handle None values properly
            skills_text = None
            if candidate.skills:
                # Truncate for list view, but keep full text available
                skills_text = candidate.skills[:200] if len(candidate.skills) > 200 else candidate.skills
            
            candidate_list.append(
                CandidateListItem(
                    candidate_id=str(candidate.candidate_id),
                    name=candidate.name if candidate.name else None,
                    email=candidate.email if candidate.email else None,
                    skills=skills_text
                )
            )
        
        return CandidateListResponse(
            total=len(candidate_list),
            candidates=candidate_list
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching candidates: {str(e)}"
        )


@router.get("/candidates/{candidate_id}", response_model=CandidateListItem)
async def get_candidate(
    candidate_id: str,
    db: Session = Depends(get_db)
):
    """Get full details of a specific candidate by ID."""
    try:
        from src.database.repository import EmbeddingRepository
        
        repository = EmbeddingRepository(db)
        candidate = repository.get_candidate_embedding(candidate_id)
        
        if not candidate:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Candidate {candidate_id} not found"
            )
        
        # Return full skills, handle None properly
        skills_value = candidate.skills if candidate.skills else None
        
        return CandidateListItem(
            candidate_id=str(candidate.candidate_id),
            name=candidate.name if candidate.name else None,
            email=candidate.email if candidate.email else None,
            skills=skills_value
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching candidate: {str(e)}"
        )


@router.post("/candidates", response_model=APIResponse)
async def create_candidate(
    request: Request,
    candidate: CandidateCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Tạo CV mới và tự động generate embeddings + recommend.
    
    Flow:
    1. Validate input
    2. Generate embeddings (Title, Skills, Experience)
    3. Lưu vào DB
    4. Background task: Update FAISS và pre-compute recommendations
    """
    try:
        from src.services.multi_field_embedding_service import MultiFieldEmbeddingService
        from src.database.multi_field_repository import MultiFieldEmbeddingRepository
        
        # Validate: experience is required
        if not candidate.experience or not candidate.experience.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Experience is required"
            )
        
        # Initialize services
        embedding_service = MultiFieldEmbeddingService(db)
        generator = embedding_service.generator
        
        # Generate embeddings
        logger.info(f"Generating embeddings for candidate {candidate.candidate_id}")
        embeddings = generator.generate_candidate_embeddings(
            title=candidate.title,
            skills=candidate.skills,
            experience=candidate.experience
        )
        
        # Check if candidate already exists
        repository = MultiFieldEmbeddingRepository(db)
        existing = repository.get_candidate_multi_embedding(candidate.candidate_id)
        is_update = existing is not None
        
        # Save to database
        candidate_record = repository.create_candidate_multi_embedding(
            candidate_id=candidate.candidate_id,
            title=candidate.title,
            skills=candidate.skills,
            experience=candidate.experience,
            title_embedding=embeddings['title_embedding'],
            skills_embedding=embeddings['skills_embedding'],
            experience_embedding=embeddings['experience_embedding'],
            name=candidate.name,
            email=candidate.email,
            replace_existing=True
        )
        
        logger.info(f"✓ Saved candidate {candidate.candidate_id} to database")
        
        # Add background task để update FAISS và pre-compute recommendations
        background_tasks.add_task(
            update_faiss_and_recommend,
            candidate_id=candidate.candidate_id,
            db=db
        )
        
        status_msg = "updated" if is_update else "created"
        
        return APIResponse(
            success=True,
            data={
                "candidate_id": candidate_record.candidate_id,
                "status": status_msg,
                "embeddings_generated": True,
                "recommendations_pending": True
            },
            message=f"Candidate {status_msg} successfully. Embeddings generated. Recommendations are being processed in background.",
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating candidate: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating candidate: {str(e)}"
        )


@router.post("/candidates/batch", response_model=APIResponse)
async def create_candidates_batch(
    request: Request,
    candidates: List[CandidateCreateRequest],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Tạo nhiều CV cùng lúc (batch create).
    
    Flow:
    1. Validate all candidates
    2. Generate embeddings cho tất cả
    3. Lưu vào DB (batch)
    4. Background task: Update FAISS và pre-compute recommendations
    """
    try:
        if not candidates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one candidate is required"
            )
        
        if len(candidates) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 100 candidates per batch"
            )
        
        from src.services.multi_field_embedding_service import MultiFieldEmbeddingService
        from src.database.multi_field_repository import MultiFieldEmbeddingRepository
        
        embedding_service = MultiFieldEmbeddingService(db)
        generator = embedding_service.generator
        repository = MultiFieldEmbeddingRepository(db)
        
        created_candidates = []
        failed = []
        
        # Process each candidate
        for candidate in candidates:
            try:
                # Validate
                if not candidate.experience or not candidate.experience.strip():
                    failed.append({
                        "candidate_id": candidate.candidate_id,
                        "error": "Experience is required"
                    })
                    continue
                
                # Generate embeddings
                embeddings = generator.generate_candidate_embeddings(
                    title=candidate.title,
                    skills=candidate.skills,
                    experience=candidate.experience
                )
                
                # Save to database
                candidate_record = repository.create_candidate_multi_embedding(
                    candidate_id=candidate.candidate_id,
                    title=candidate.title,
                    skills=candidate.skills,
                    experience=candidate.experience,
                    title_embedding=embeddings['title_embedding'],
                    skills_embedding=embeddings['skills_embedding'],
                    experience_embedding=embeddings['experience_embedding'],
                    name=candidate.name,
                    email=candidate.email,
                    replace_existing=True
                )
                
                created_candidates.append(candidate_record.candidate_id)
                
            except Exception as e:
                logger.error(f"Error processing candidate {candidate.candidate_id}: {e}")
                failed.append({
                    "candidate_id": candidate.candidate_id,
                    "error": str(e)
                })
        
        # Add background task để update FAISS và pre-compute recommendations
        if created_candidates:
            background_tasks.add_task(
                update_faiss_for_batch,
                candidate_ids=created_candidates,
                db=db
            )
        
        return APIResponse(
            success=True,
            data={
                "total_requested": len(candidates),
                "created": len(created_candidates),
                "failed": len(failed),
                "candidate_ids": created_candidates,
                "failed_candidates": failed
            },
            message=f"Batch processing completed: {len(created_candidates)} created, {len(failed)} failed. Recommendations are being processed in background.",
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch create candidates: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating candidates batch: {str(e)}"
        )


@router.post("/match/candidate-id", response_model=JobMatchResponse)
async def match_candidate_by_id(
    request: CandidateIdRequest,
    db: Session = Depends(get_db)
):
    """Find top matching jobs for a processed candidate using their existing vector from PostgreSQL (up to 10k results)."""
    try:
        from src.services.matching_service import MatchingService
        
        matching_service = MatchingService(db, use_faiss=True, use_reranking=True)
        
        # Find matching jobs using candidate_id (vector from PostgreSQL)
        matches = matching_service.find_jobs_for_candidate(
            candidate_id=request.candidate_id,
            top_k=request.limit
        )
        
        job_matches = [
            JobMatchInfo(
                job_id=match["job_id"],
                title=match["title"],
                company=match.get("company"),
                location=match.get("location"),
                description=match.get("description"),
                requirements=match.get("requirements"),
                similarity_score=match["similarity_score"]
            )
            for match in matches
        ]
        
        return JobMatchResponse(
            candidate_id=request.candidate_id,
            total_matches=len(job_matches),
            matches=job_matches
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error matching candidate: {str(e)}"
        )


@router.post("/match/candidate", response_model=JobMatchResponse)
async def match_candidate(
    request: CandidateInputRequest,
    db: Session = Depends(get_db)
):
    """Find top matching jobs for a new candidate with detailed input (up to 10k results)."""
    try:
        from src.services.matching_service import MatchingService
        
        matching_service = MatchingService(db, use_faiss=True, use_reranking=True)
        
        # Combine candidate fields into text
        candidate_text = matching_service.combine_candidate_fields(
            name=request.name,
            skills=request.skills,
            experience=request.experience,
            education=request.education,
            summary=request.summary,
            resume_text=request.resume_text
        )
        
        if not candidate_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one candidate field (skills, experience, summary, education, or resume_text) must be provided"
            )
        
        # Find matching jobs (will embed the text, then match using vectors from PostgreSQL)
        matches = matching_service.find_jobs_for_candidate_text(
            candidate_text=candidate_text,
            top_k=request.limit
        )
        
        job_matches = [
            JobMatchInfo(
                job_id=match["job_id"],
                title=match["title"],
                company=match.get("company"),
                location=match.get("location"),
                description=match.get("description"),
                requirements=match.get("requirements"),
                similarity_score=match["similarity_score"]
            )
            for match in matches
        ]
        
        return JobMatchResponse(
            total_matches=len(job_matches),
            matches=job_matches
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error matching candidate: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "job-recommendation-api"}


@router.post("/jobs/ids", response_model=JobIdsResponse)
async def get_job_ids_for_candidate(
    request: CandidateIdRequest,
    db: Session = Depends(get_db)
):
    """
    Get top 10 job IDs for a candidate from processed recommendations.
    This is a fast endpoint that only returns job IDs without embedding computation.
    """
    try:
        from src.services.matching_service import MatchingService
        
        matching_service = MatchingService(db, use_faiss=True, use_reranking=True)
        
        # Get job IDs directly from processed recommendations
        job_ids = matching_service.get_job_ids_for_candidate(
            candidate_id=request.candidate_id,
            top_k=min(request.limit, 10)  # Max 10 from processed table
        )
        
        return JobIdsResponse(
            candidate_id=request.candidate_id,
            job_ids=job_ids,
            total=len(job_ids)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting job IDs: {str(e)}"
        )


@router.get("/scheduler/status")
async def get_scheduler_status():
    """Get status of scheduled jobs."""
    try:
        from src.api.main import get_scheduler_service
        scheduler_service = get_scheduler_service()
        status = scheduler_service.get_job_status()
        return status
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting scheduler status: {str(e)}"
        )


@router.post("/scheduler/precompute")
async def trigger_precompute(db: Session = Depends(get_db)):
    """Manually trigger pre-computation of recommendations for all candidates."""
    try:
        from src.services.precompute_service import PrecomputeService
        
        precompute_service = PrecomputeService(db)
        results = precompute_service.precompute_all_candidates(top_k=10)
        
        return {
            "message": "Pre-computation completed",
            "results": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error triggering pre-computation: {str(e)}"
        )


# ============================================================================
# Multi-Field Filter Matching Routes
# ============================================================================

@router.post("/multi-filter/recommend/jobs", response_model=JobRecommendationResponse)
async def multi_filter_recommend_jobs(
    request: JobRecommendationRequest,
    db: Session = Depends(get_db)
):
    """Get job recommendations using multi-field filter pipeline (experience → skills → title)."""
    try:
        from src.services.multi_filter_matching_service import MultiFilterMatchingService
        
        matching_service = MultiFilterMatchingService(db, use_faiss=True)
        recommendations = matching_service.find_jobs_for_candidate(
            candidate_id=request.candidate_id,
            top_k=request.limit
        )
        
        job_infos = [
            JobInfo(
                job_id=job["job_id"],
                title=job["title"],
                company=job.get("company"),
                location=job.get("location"),
                description=None  # Multi-field doesn't store description
            )
            for job in recommendations
        ]
        
        return JobRecommendationResponse(
            candidate_id=request.candidate_id,
            recommendations=job_infos
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting multi-filter job recommendations: {str(e)}"
        )


@router.post("/multi-filter/match/candidate")
async def multi_filter_match_candidate(
    request: CandidateInputRequest,
    db: Session = Depends(get_db)
):
    """Find matching jobs using multi-field filter pipeline for new candidate."""
    try:
        from src.services.multi_filter_matching_service import MultiFilterMatchingService
        
        matching_service = MultiFilterMatchingService(db, use_faiss=True)
        matches = matching_service.find_jobs_for_candidate_text(
            title=None,  # Can extract from other fields if needed
            skills=request.skills,
            experience=request.experience,
            top_k=request.limit
        )
        
        job_matches = [
            JobMatchInfo(
                job_id=match["job_id"],
                title=match["title"],
                company=match.get("company"),
                location=match.get("location"),
                description=None,
                requirements=None,
                similarity_score=match["similarity_score"]
            )
            for match in matches
        ]
        
        return JobMatchResponse(
            total_matches=len(job_matches),
            matches=job_matches
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error matching candidate with multi-filter: {str(e)}"
        )


@router.post("/multi-filter/process/jd-dataset", response_model=DatasetProcessResponse)
async def multi_filter_process_jd_dataset(
    request: DatasetProcessRequest,
    db: Session = Depends(get_db)
):
    """Process JD dataset with multi-field embeddings (title, skills, requirements)."""
    try:
        from src.services.multi_field_embedding_service import MultiFieldEmbeddingService
        
        service = MultiFieldEmbeddingService(db)
        count = service.process_jd_dataset(request.file_path, request.file_type)
        return DatasetProcessResponse(
            message="JD dataset processed successfully with multi-field embeddings",
            records_processed=count
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing JD dataset with multi-field: {str(e)}"
        )


@router.post("/multi-filter/process/candidate-dataset", response_model=DatasetProcessResponse)
async def multi_filter_process_candidate_dataset(
    request: DatasetProcessRequest,
    db: Session = Depends(get_db)
):
    """Process candidate dataset with multi-field embeddings (title, skills, experience)."""
    try:
        from src.services.multi_field_embedding_service import MultiFieldEmbeddingService
        
        service = MultiFieldEmbeddingService(db)
        count = service.process_candidate_dataset(request.file_path, request.file_type)
        return DatasetProcessResponse(
            message="Candidate dataset processed successfully with multi-field embeddings",
            records_processed=count
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing candidate dataset with multi-field: {str(e)}"
        )

