"""Pydantic schemas for Two-Tower API."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, List


class FieldWeights(BaseModel):
    """Field weights for matching."""
    title: float = Field(default=0.2, ge=0.0, le=1.0)
    skills: float = Field(default=0.4, ge=0.0, le=1.0)
    experience: float = Field(default=0.4, ge=0.0, le=1.0)


class JobSearchRequest(BaseModel):
    """Request for searching jobs for a candidate."""
    candidate_id: str
    top_k: int = Field(default=10, ge=1, le=100)
    weights: Optional[FieldWeights] = None
    use_reranking: bool = False


class CandidateSearchRequest(BaseModel):
    """Request for searching candidates for a job."""
    job_id: str
    top_k: int = Field(default=10, ge=1, le=100)
    weights: Optional[FieldWeights] = None


class FieldScores(BaseModel):
    """Per-field similarity scores."""
    title: float
    skills: float
    experience: float


class JobMatch(BaseModel):
    """Job match result."""
    job_id: str
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    score: float
    explain: FieldScores


class CandidateMatch(BaseModel):
    """Candidate match result."""
    candidate_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    score: float
    explain: FieldScores


class JobSearchResponse(BaseModel):
    """Response for job search."""
    total_matches: int
    matches: List[JobMatch]


class CandidateSearchResponse(BaseModel):
    """Response for candidate search."""
    total_matches: int
    matches: List[CandidateMatch]


class IndexJobRequest(BaseModel):
    """Request for indexing a job."""
    job_id: str
    title: str
    skills: Optional[str] = None
    requirement: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None


class IndexCandidateRequest(BaseModel):
    """Request for indexing a candidate."""
    candidate_id: str
    title: Optional[str] = None
    skills: Optional[str] = None
    experience: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None


class IndexResponse(BaseModel):
    """Response for indexing."""
    status: str
    job_id: Optional[str] = None
    candidate_id: Optional[str] = None
    message: str


class ReindexRequest(BaseModel):
    """Request for reindexing."""
    reindex_type: str = Field(default="full", pattern="^(full|incremental|job|candidate)$")
    force: bool = False


class ReindexResponse(BaseModel):
    """Response for reindex."""
    status: str
    reindex_id: Optional[int] = None
    message: str
    estimated_time_minutes: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    indices: Dict[str, str]
    database: str
    total_jobs: int
    total_candidates: int


