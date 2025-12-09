"""Two-Tower FAISS index manager for 6 separate indices (3 per tower)."""
import faiss
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from sqlalchemy.orm import Session
import logging
from pathlib import Path
import pickle

from src.database.models import JobDescriptionTwoTower, CandidateTwoTower

logger = logging.getLogger(__name__)


class TwoTowerFAISSManager:
    """Manages 6 separate FAISS indices for Two-Tower architecture."""
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "HNSW",
        index_params: Optional[Dict[str, Any]] = None,
        normalize: bool = True
    ):
        """
        Initialize Two-Tower FAISS index manager.
        
        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index ("Flat", "IVF", "HNSW")
            index_params: Parameters for index construction
            normalize: Whether to normalize vectors (for cosine similarity)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.normalize = normalize
        self.index_params = index_params or {}
        
        # 3 indices for Jobs (Job Tower)
        self.job_title_index: Optional[faiss.Index] = None
        self.job_skills_index: Optional[faiss.Index] = None
        self.job_requirement_index: Optional[faiss.Index] = None
        
        # 3 indices for Candidates (Candidate Tower)
        self.candidate_title_index: Optional[faiss.Index] = None
        self.candidate_skills_index: Optional[faiss.Index] = None
        self.candidate_experience_index: Optional[faiss.Index] = None
        
        # ID maps: FAISS index position -> id
        self.job_title_id_map: Dict[int, str] = {}
        self.job_skills_id_map: Dict[int, str] = {}
        self.job_requirement_id_map: Dict[int, str] = {}
        self.candidate_title_id_map: Dict[int, str] = {}
        self.candidate_skills_id_map: Dict[int, str] = {}
        self.candidate_experience_id_map: Dict[int, str] = {}
        
        # Reverse maps: id -> FAISS index position
        self.job_title_reverse_map: Dict[str, int] = {}
        self.job_skills_reverse_map: Dict[str, int] = {}
        self.job_requirement_reverse_map: Dict[str, int] = {}
        self.candidate_title_reverse_map: Dict[str, int] = {}
        self.candidate_skills_reverse_map: Dict[str, int] = {}
        self.candidate_experience_reverse_map: Dict[str, int] = {}
        
        self._initialize_indices()
    
    def _initialize_indices(self):
        """Initialize FAISS indices based on type."""
        if self.index_type == "HNSW":
            M = self.index_params.get('M', 32)
            
            # Job indices
            self.job_title_index = faiss.IndexHNSWFlat(self.dimension, M)
            self.job_skills_index = faiss.IndexHNSWFlat(self.dimension, M)
            self.job_requirement_index = faiss.IndexHNSWFlat(self.dimension, M)
            
            # Candidate indices
            self.candidate_title_index = faiss.IndexHNSWFlat(self.dimension, M)
            self.candidate_skills_index = faiss.IndexHNSWFlat(self.dimension, M)
            self.candidate_experience_index = faiss.IndexHNSWFlat(self.dimension, M)
            
            # Set ef_construction
            ef_construction = self.index_params.get('ef_construction', 200)
            for index in [self.job_title_index, self.job_skills_index, self.job_requirement_index,
                         self.candidate_title_index, self.candidate_skills_index, self.candidate_experience_index]:
                index.hnsw.efConstruction = ef_construction
        elif self.index_type == "Flat":
            self.job_title_index = faiss.IndexFlatL2(self.dimension)
            self.job_skills_index = faiss.IndexFlatL2(self.dimension)
            self.job_requirement_index = faiss.IndexFlatL2(self.dimension)
            self.candidate_title_index = faiss.IndexFlatL2(self.dimension)
            self.candidate_skills_index = faiss.IndexFlatL2(self.dimension)
            self.candidate_experience_index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return vectors / norms
        return vectors
    
    def build_indices_from_db(self, db: Session, batch_size: int = 1000):
        """Build 6 FAISS indices from PostgreSQL database."""
        logger.info("Building Two-Tower FAISS indices from database...")
        
        # Build job indices
        all_jobs = db.query(JobDescriptionTwoTower).all()
        total_jobs = len(all_jobs)
        
        if total_jobs > 0:
            logger.info(f"Found {total_jobs} jobs. Building 3 job indices...")
            title_vectors = []
            skills_vectors = []
            requirement_vectors = []
            
            for i, job in enumerate(all_jobs):
                title_vectors.append(np.array(job.title_embedding, dtype=np.float32))
                skills_vectors.append(np.array(job.skills_embedding, dtype=np.float32))
                requirement_vectors.append(np.array(job.requirement_embedding, dtype=np.float32))
                
                # Store ID mappings
                self.job_title_id_map[i] = job.job_id
                self.job_skills_id_map[i] = job.job_id
                self.job_requirement_id_map[i] = job.job_id
                
                self.job_title_reverse_map[job.job_id] = i
                self.job_skills_reverse_map[job.job_id] = i
                self.job_requirement_reverse_map[job.job_id] = i
            
            # Convert to numpy arrays and normalize
            title_matrix = np.vstack(title_vectors)
            skills_matrix = np.vstack(skills_vectors)
            requirement_matrix = np.vstack(requirement_vectors)
            
            title_matrix = self._normalize_vectors(title_matrix)
            skills_matrix = self._normalize_vectors(skills_matrix)
            requirement_matrix = self._normalize_vectors(requirement_matrix)
            
            # Add vectors to indices
            self.job_title_index.add(title_matrix)
            self.job_skills_index.add(skills_matrix)
            self.job_requirement_index.add(requirement_matrix)
            
            logger.info(f"✓ Built 3 job indices with {total_jobs} jobs each")
        else:
            logger.warning("No jobs found in database")
        
        # Build candidate indices
        all_candidates = db.query(CandidateTwoTower).all()
        total_candidates = len(all_candidates)
        
        if total_candidates > 0:
            logger.info(f"Found {total_candidates} candidates. Building 3 candidate indices...")
            title_vectors = []
            skills_vectors = []
            experience_vectors = []
            
            for i, candidate in enumerate(all_candidates):
                title_vectors.append(np.array(candidate.title_embedding, dtype=np.float32))
                skills_vectors.append(np.array(candidate.skills_embedding, dtype=np.float32))
                experience_vectors.append(np.array(candidate.experience_embedding, dtype=np.float32))
                
                # Store ID mappings
                self.candidate_title_id_map[i] = candidate.candidate_id
                self.candidate_skills_id_map[i] = candidate.candidate_id
                self.candidate_experience_id_map[i] = candidate.candidate_id
                
                self.candidate_title_reverse_map[candidate.candidate_id] = i
                self.candidate_skills_reverse_map[candidate.candidate_id] = i
                self.candidate_experience_reverse_map[candidate.candidate_id] = i
            
            # Convert to numpy arrays and normalize
            title_matrix = np.vstack(title_vectors)
            skills_matrix = np.vstack(skills_vectors)
            experience_matrix = np.vstack(experience_vectors)
            
            title_matrix = self._normalize_vectors(title_matrix)
            skills_matrix = self._normalize_vectors(skills_matrix)
            experience_matrix = self._normalize_vectors(experience_matrix)
            
            # Add vectors to indices
            self.candidate_title_index.add(title_matrix)
            self.candidate_skills_index.add(skills_matrix)
            self.candidate_experience_index.add(experience_matrix)
            
            logger.info(f"✓ Built 3 candidate indices with {total_candidates} candidates each")
        else:
            logger.warning("No candidates found in database")
    
    def search_job_by_field(
        self,
        query_embedding: List[float],
        field_type: str,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for similar jobs by field type.
        
        Args:
            query_embedding: Query vector
            field_type: 'title', 'skills', or 'requirement'
            k: Number of results to return
        
        Returns:
            List of (job_id, similarity_score) tuples
        """
        # Select appropriate index and ID map
        if field_type == 'title':
            index = self.job_title_index
            id_map = self.job_title_id_map
        elif field_type == 'skills':
            index = self.job_skills_index
            id_map = self.job_skills_id_map
        elif field_type == 'requirement':
            index = self.job_requirement_index
            id_map = self.job_requirement_id_map
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        if index is None or index.ntotal == 0:
            logger.warning(f"Index for job field '{field_type}' is empty")
            return []
        
        # Prepare query vector
        query_vec = np.array([query_embedding], dtype=np.float32)
        query_vec = self._normalize_vectors(query_vec)
        
        # Set ef_search for HNSW
        if self.index_type == "HNSW":
            ef_search = self.index_params.get('ef_search', 128)
            index.hnsw.efSearch = ef_search
        
        # Search
        distances, indices = index.search(query_vec, min(k, index.ntotal))
        
        # Convert to results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # Invalid index
                continue
            job_id = id_map.get(idx)
            if job_id:
                # Convert L2 distance to cosine similarity (since vectors are normalized)
                similarity = 1.0 - (dist / 2.0)  # L2 distance -> cosine similarity
                similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
                results.append((job_id, float(similarity)))
        
        return results
    
    def search_candidate_by_field(
        self,
        query_embedding: List[float],
        field_type: str,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for similar candidates by field type.
        
        Args:
            query_embedding: Query vector
            field_type: 'title', 'skills', or 'experience'
            k: Number of results to return
        
        Returns:
            List of (candidate_id, similarity_score) tuples
        """
        # Select appropriate index and ID map
        if field_type == 'title':
            index = self.candidate_title_index
            id_map = self.candidate_title_id_map
        elif field_type == 'skills':
            index = self.candidate_skills_index
            id_map = self.candidate_skills_id_map
        elif field_type == 'experience':
            index = self.candidate_experience_index
            id_map = self.candidate_experience_id_map
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        if index is None or index.ntotal == 0:
            logger.warning(f"Index for candidate field '{field_type}' is empty")
            return []
        
        # Prepare query vector
        query_vec = np.array([query_embedding], dtype=np.float32)
        query_vec = self._normalize_vectors(query_vec)
        
        # Set ef_search for HNSW
        if self.index_type == "HNSW":
            ef_search = self.index_params.get('ef_search', 128)
            index.hnsw.efSearch = ef_search
        
        # Search
        distances, indices = index.search(query_vec, min(k, index.ntotal))
        
        # Convert to results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:
                continue
            candidate_id = id_map.get(idx)
            if candidate_id:
                similarity = 1.0 - (dist / 2.0)
                similarity = max(0.0, min(1.0, similarity))
                results.append((candidate_id, float(similarity)))
        
        return results
    
    def save_indices(self, base_path: Path):
        """Save all indices to disk."""
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save job indices
        faiss.write_index(self.job_title_index, str(base_path / "job_title_index.faiss"))
        faiss.write_index(self.job_skills_index, str(base_path / "job_skills_index.faiss"))
        faiss.write_index(self.job_requirement_index, str(base_path / "job_requirement_index.faiss"))
        
        # Save candidate indices
        faiss.write_index(self.candidate_title_index, str(base_path / "candidate_title_index.faiss"))
        faiss.write_index(self.candidate_skills_index, str(base_path / "candidate_skills_index.faiss"))
        faiss.write_index(self.candidate_experience_index, str(base_path / "candidate_experience_index.faiss"))
        
        # Save ID maps
        with open(base_path / "job_title_index.pkl", "wb") as f:
            pickle.dump(self.job_title_id_map, f)
        with open(base_path / "job_skills_index.pkl", "wb") as f:
            pickle.dump(self.job_skills_id_map, f)
        with open(base_path / "job_requirement_index.pkl", "wb") as f:
            pickle.dump(self.job_requirement_id_map, f)
        with open(base_path / "candidate_title_index.pkl", "wb") as f:
            pickle.dump(self.candidate_title_id_map, f)
        with open(base_path / "candidate_skills_index.pkl", "wb") as f:
            pickle.dump(self.candidate_skills_id_map, f)
        with open(base_path / "candidate_experience_index.pkl", "wb") as f:
            pickle.dump(self.candidate_experience_id_map, f)
        
        logger.info(f"✓ Saved all indices to {base_path}")
    
    def load_indices(self, base_path: Path):
        """Load all indices from disk."""
        # Load job indices
        self.job_title_index = faiss.read_index(str(base_path / "job_title_index.faiss"))
        self.job_skills_index = faiss.read_index(str(base_path / "job_skills_index.faiss"))
        self.job_requirement_index = faiss.read_index(str(base_path / "job_requirement_index.faiss"))
        
        # Load candidate indices
        self.candidate_title_index = faiss.read_index(str(base_path / "candidate_title_index.faiss"))
        self.candidate_skills_index = faiss.read_index(str(base_path / "candidate_skills_index.faiss"))
        self.candidate_experience_index = faiss.read_index(str(base_path / "candidate_experience_index.faiss"))
        
        # Load ID maps
        with open(base_path / "job_title_index.pkl", "rb") as f:
            self.job_title_id_map = pickle.load(f)
        with open(base_path / "job_skills_index.pkl", "rb") as f:
            self.job_skills_id_map = pickle.load(f)
        with open(base_path / "job_requirement_index.pkl", "rb") as f:
            self.job_requirement_id_map = pickle.load(f)
        with open(base_path / "candidate_title_index.pkl", "rb") as f:
            self.candidate_title_id_map = pickle.load(f)
        with open(base_path / "candidate_skills_index.pkl", "rb") as f:
            self.candidate_skills_id_map = pickle.load(f)
        with open(base_path / "candidate_experience_index.pkl", "rb") as f:
            self.candidate_experience_id_map = pickle.load(f)
        
        # Rebuild reverse maps
        self.job_title_reverse_map = {v: k for k, v in self.job_title_id_map.items()}
        self.job_skills_reverse_map = {v: k for k, v in self.job_skills_id_map.items()}
        self.job_requirement_reverse_map = {v: k for k, v in self.job_requirement_id_map.items()}
        self.candidate_title_reverse_map = {v: k for k, v in self.candidate_title_id_map.items()}
        self.candidate_skills_reverse_map = {v: k for k, v in self.candidate_skills_id_map.items()}
        self.candidate_experience_reverse_map = {v: k for k, v in self.candidate_experience_id_map.items()}
        
        logger.info(f"✓ Loaded all indices from {base_path}")


