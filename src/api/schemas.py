"""Pydantic schemas for API requests and responses."""
from pydantic import BaseModel, Field
from typing import Optional, List


class JobRecommendationRequest(BaseModel):
    """Request schema for job recommendations."""
    candidate_id: str = Field(..., description="Candidate ID")
    limit: int = Field(default=50, ge=1, le=100, description="Number of recommendations (default: 50)")


class CandidateRecommendationRequest(BaseModel):
    """Request schema for candidate recommendations."""
    job_id: str = Field(..., description="Job ID")
    limit: int = Field(default=10, ge=1, le=100, description="Number of recommendations")


class JobInfo(BaseModel):
    """Job information schema."""
    job_id: str
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None


class CandidateInfo(BaseModel):
    """Candidate information schema."""
    candidate_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    skills: Optional[str] = None
    summary: Optional[str] = None


class JobRecommendationResponse(BaseModel):
    """Response schema for job recommendations."""
    candidate_id: str
    recommendations: List[JobInfo]


class CandidateRecommendationResponse(BaseModel):
    """Response schema for candidate recommendations."""
    job_id: str
    recommendations: List[CandidateInfo]


class CandidateTextRequest(BaseModel):
    """Request schema for candidate text matching."""
    candidate_text: str = Field(..., description="Candidate information as text")
    limit: int = Field(default=50, ge=1, le=10000, description="Number of top matches (up to 10k)")

class CandidateInputRequest(BaseModel):
    """Request schema for detailed candidate input."""
    name: Optional[str] = Field(None, description="Candidate name")
    skills: Optional[str] = Field(None, description="Skills and competencies")
    experience: Optional[str] = Field(None, description="Work experience")
    education: Optional[str] = Field(None, description="Education background")
    summary: Optional[str] = Field(None, description="Professional summary")
    resume_text: Optional[str] = Field(None, description="Full resume text")
    limit: int = Field(default=15, ge=1, le=10000, description="Number of top matches (default: 15)")

class CandidateIdRequest(BaseModel):
    """Request schema for matching using candidate_id from database."""
    candidate_id: str = Field(..., description="Candidate ID from database")
    limit: int = Field(default=15, ge=1, le=10000, description="Number of top matches (default: 15)")

class CandidateListItem(BaseModel):
    """Schema for candidate list item."""
    candidate_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    skills: Optional[str] = None

class CandidateListResponse(BaseModel):
    """Response schema for candidate list."""
    total: int
    candidates: List[CandidateListItem]


class CandidateFileRequest(BaseModel):
    """Request schema for candidate file matching."""
    candidate_file: str = Field(..., description="Path to processed candidate dataset")
    candidate_index: int = Field(default=0, ge=0, description="Index of candidate in file (0-based)")
    limit: int = Field(default=50, ge=1, le=100, description="Number of top matches")


class JobMatchInfo(BaseModel):
    """Job match information with similarity score."""
    job_id: str
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    requirements: Optional[str] = None
    similarity_score: float = Field(..., description="Similarity score (0-1)")


class JobMatchResponse(BaseModel):
    """Response schema for job matches."""
    candidate_id: Optional[str] = None
    candidate_index: Optional[int] = None
    total_matches: int
    matches: List[JobMatchInfo]


class DatasetProcessRequest(BaseModel):
    """Request schema for processing datasets."""
    file_path: str = Field(..., description="Path to the dataset file")
    file_type: str = Field(default="csv", description="File type: csv or json")


class DatasetProcessResponse(BaseModel):
    """Response schema for dataset processing."""
    message: str
    records_processed: int


class JobIdsResponse(BaseModel):
    """Response schema for job IDs only (fast query)."""
    candidate_id: str
    job_ids: List[str] = Field(..., description="List of top 10 job IDs")
    total: int = Field(..., description="Total number of job IDs returned")
