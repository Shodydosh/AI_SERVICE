"""SQLAlchemy models for main database tables."""
from sqlalchemy import Column, String, Boolean, Integer, Float, Date, DateTime, Text, ForeignKey, ARRAY
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class CV(Base):
    """CV model."""
    __tablename__ = "cvs"
    
    id = Column(String(255), primary_key=True)
    userId = Column(String(255), nullable=True)  # Removed FK constraint to allow import without users table
    title = Column(String(255), nullable=True)
    isMain = Column(Boolean, nullable=False, default=False)
    fullName = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    phoneNumber = Column(String(255), nullable=True)
    dateOfBirth = Column(Date, nullable=True)
    gender = Column(Integer, nullable=True)
    address = Column(String(255), nullable=True)
    currentPosition = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    objective = Column(Text, nullable=True)
    lastGeneratedAt = Column(DateTime, nullable=True)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    isOpenForJob = Column(Boolean, nullable=False, default=True)
    
    # Embedding columns (float4[])
    title_embedding = Column(PG_ARRAY(Float(precision=53, asdecimal=False)), nullable=True)
    skills_embedding = Column(PG_ARRAY(Float(precision=53, asdecimal=False)), nullable=True)
    experience_embedding = Column(PG_ARRAY(Float(precision=53, asdecimal=False)), nullable=True)
    contentHash = Column(String(255), nullable=True)
    
    # Relationships
    skills = relationship("CVSkill", back_populates="cv", lazy="joined")
    work_experiences = relationship("WorkExperience", back_populates="cv", lazy="joined")


class WorkExperience(Base):
    """WorkExperience model."""
    __tablename__ = "work_experiences"
    
    id = Column(String(255), primary_key=True)
    cvId = Column(String(255), ForeignKey("cvs.id"), nullable=False)
    title = Column(String(255), nullable=True)
    company = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    startDate = Column(Date, nullable=True)
    endDate = Column(Date, nullable=True)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    cv = relationship("CV", back_populates="work_experiences")


class CVSkill(Base):
    """CVSkill model."""
    __tablename__ = "cv_skills"
    
    id = Column(String(255), primary_key=True)
    cvId = Column(String(255), ForeignKey("cvs.id"), nullable=False)
    skillName = Column(String(255), nullable=True)
    level = Column(Integer, nullable=True)
    yearsOfExperience = Column(Integer, nullable=True)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    cv = relationship("CV", back_populates="skills")


class Job(Base):
    """Job model."""
    __tablename__ = "jobs"
    
    id = Column(String(255), primary_key=True)
    companyId = Column(String(255), nullable=True)  # Removed FK constraint to allow import without companies table
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    location = Column(String(255), nullable=True)
    industry = Column(String(255), nullable=True)
    experienceLevel = Column(Integer, nullable=True)
    type = Column(Integer, nullable=True)
    urgent = Column(Boolean, nullable=False, default=False)
    expiresAt = Column(DateTime, nullable=True)
    applicationCount = Column(Integer, nullable=False, default=0)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(50), nullable=True)  # Changed to String to avoid enum issues during import
    
    # Embedding columns (float4[])
    title_embedding = Column(PG_ARRAY(Float(precision=53, asdecimal=False)), nullable=True)
    skills_embedding = Column(PG_ARRAY(Float(precision=53, asdecimal=False)), nullable=True)
    requirement_embedding = Column(PG_ARRAY(Float(precision=53, asdecimal=False)), nullable=True)
    contentHash = Column(String(255), nullable=True)
    
    # Relationships
    requirements = relationship("JobRequirement", back_populates="job", lazy="joined")


class JobRequirement(Base):
    """JobRequirement model."""
    __tablename__ = "job_requirements"
    
    id = Column(String(255), primary_key=True)
    jobId = Column(String(255), ForeignKey("jobs.id"), nullable=False)
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    createdAt = Column(DateTime, nullable=False, default=datetime.utcnow)
    updatedAt = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    job = relationship("Job", back_populates="requirements")
    
    @property
    def requirement(self):
        """Get requirement text from title and description."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.description:
            parts.append(self.description)
        return "\n".join(parts) if parts else None
