"""Multi-field FAISS index manager for 3 separate embeddings per record."""
import faiss
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from sqlalchemy.orm import Session
import logging
from pathlib import Path
import pickle

from src.database.models import JobDescriptionMultiEmbedding, CandidateMultiEmbedding

logger = logging.getLogger(__name__)


class MultiFieldFAISSManager:
    """Manages 3 separate FAISS indices for multi-field embeddings."""
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "HNSW",
        index_params: Optional[Dict[str, Any]] = None,
        normalize: bool = True
    ):
        """
        Initialize multi-field FAISS index manager.
        
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
        
        # 3 indices for jobs: title, skills, requirement
        self.jd_title_index: Optional[faiss.Index] = None
        self.jd_skills_index: Optional[faiss.Index] = None
        self.jd_requirement_index: Optional[faiss.Index] = None
        
        # ID maps: FAISS index position -> job_id
        self.jd_title_id_map: Dict[int, str] = {}
        self.jd_skills_id_map: Dict[int, str] = {}
        self.jd_requirement_id_map: Dict[int, str] = {}
        
        # Reverse map: job_id -> FAISS index position
        self.jd_title_reverse_map: Dict[str, int] = {}
        self.jd_skills_reverse_map: Dict[str, int] = {}
        self.jd_requirement_reverse_map: Dict[str, int] = {}
        
        self._initialize_indices()
    
    def _initialize_indices(self):
        """Initialize FAISS indices based on type."""
        if self.index_type == "HNSW":
            M = self.index_params.get('M', 32)
            
            self.jd_title_index = faiss.IndexHNSWFlat(self.dimension, M)
            self.jd_skills_index = faiss.IndexHNSWFlat(self.dimension, M)
            self.jd_requirement_index = faiss.IndexHNSWFlat(self.dimension, M)
            
            # Set ef_construction
            ef_construction = self.index_params.get('ef_construction', 200)
            self.jd_title_index.hnsw.efConstruction = ef_construction
            self.jd_skills_index.hnsw.efConstruction = ef_construction
            self.jd_requirement_index.hnsw.efConstruction = ef_construction
        elif self.index_type == "Flat":
            self.jd_title_index = faiss.IndexFlatL2(self.dimension)
            self.jd_skills_index = faiss.IndexFlatL2(self.dimension)
            self.jd_requirement_index = faiss.IndexFlatL2(self.dimension)
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
        """Build 3 FAISS indices from PostgreSQL database."""
        logger.info("Building multi-field FAISS indices from database...")
        
        # Get all jobs
        all_jobs = db.query(JobDescriptionMultiEmbedding).all()
        total_jobs = len(all_jobs)
        
        if total_jobs == 0:
            logger.warning("No jobs found in database")
            return
        
        logger.info(f"Found {total_jobs} jobs. Building 3 indices...")
        
        # Process in batches
        title_vectors = []
        skills_vectors = []
        requirement_vectors = []
        
        for i, job in enumerate(all_jobs):
            title_vectors.append(np.array(job.title_embedding, dtype=np.float32))
            skills_vectors.append(np.array(job.skills_embedding, dtype=np.float32))
            requirement_vectors.append(np.array(job.requirement_embedding, dtype=np.float32))
            
            # Store ID mappings
            idx = i
            self.jd_title_id_map[idx] = job.job_id
            self.jd_skills_id_map[idx] = job.job_id
            self.jd_requirement_id_map[idx] = job.job_id
            
            self.jd_title_reverse_map[job.job_id] = idx
            self.jd_skills_reverse_map[job.job_id] = idx
            self.jd_requirement_reverse_map[job.job_id] = idx
        
        # Convert to numpy arrays and normalize
        title_matrix = np.vstack(title_vectors)
        skills_matrix = np.vstack(skills_vectors)
        requirement_matrix = np.vstack(requirement_vectors)
        
        title_matrix = self._normalize_vectors(title_matrix)
        skills_matrix = self._normalize_vectors(skills_matrix)
        requirement_matrix = self._normalize_vectors(requirement_matrix)
        
        # Add vectors to indices
        logger.info("Adding vectors to title index...")
        self.jd_title_index.add(title_matrix)
        
        logger.info("Adding vectors to skills index...")
        self.jd_skills_index.add(skills_matrix)
        
        logger.info("Adding vectors to requirement index...")
        self.jd_requirement_index.add(requirement_matrix)
        
        logger.info(f"✓ Built 3 indices with {total_jobs} jobs each")
    
    def search(
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
        # Select index and ID map based on field type
        if field_type == 'title':
            index = self.jd_title_index
            id_map = self.jd_title_id_map
        elif field_type == 'skills':
            index = self.jd_skills_index
            id_map = self.jd_skills_id_map
        elif field_type == 'requirement':
            index = self.jd_requirement_index
            id_map = self.jd_requirement_id_map
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        if index is None or index.ntotal == 0:
            logger.warning(f"Index not built for {field_type}")
            return []
        
        # Prepare query vector
        query_vector = np.array([query_embedding], dtype=np.float32)
        query_vector = self._normalize_vectors(query_vector)
        
        # Set search parameters
        if self.index_type == "HNSW":
            ef_search = self.index_params.get('ef_search', 64)
            index.hnsw.efSearch = ef_search
        
        # Search
        distances, indices = index.search(query_vector, k)
        
        # Convert to (job_id, similarity) tuples
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0:  # Invalid index
                continue
            job_id = id_map.get(idx)
            if job_id:
                # Convert squared L2 distance to cosine similarity
                # FAISS returns squared L2 distance (not L2 distance)
                # For normalized vectors: cosine_similarity = 1 - (squared_L2_distance / 2)
                # Formula: ||a-b||^2 = 2 - 2(a·b) = 2(1 - cosine_sim)
                # Therefore: cosine_sim = 1 - (||a-b||^2 / 2)
                similarity = 1 - (distance / 2.0)
                similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
                results.append((job_id, similarity))
        
        return results
    
    def search_filtered(
        self,
        query_embedding: List[float],
        field_type: str,
        candidate_ids: List[str],
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Search for similar jobs by field type, filtered to candidate job IDs.
        
        Args:
            query_embedding: Query vector
            field_type: 'title', 'skills', or 'requirement'
            candidate_ids: List of job IDs to filter results to
            k: Number of results to return
        
        Returns:
            List of (job_id, similarity_score) tuples
        """
        # Select index and maps based on field type
        if field_type == 'title':
            index = self.jd_title_index
            reverse_map = self.jd_title_reverse_map
            id_map = self.jd_title_id_map
        elif field_type == 'skills':
            index = self.jd_skills_index
            reverse_map = self.jd_skills_reverse_map
            id_map = self.jd_skills_id_map
        elif field_type == 'requirement':
            index = self.jd_requirement_index
            reverse_map = self.jd_requirement_reverse_map
            id_map = self.jd_requirement_id_map
        else:
            raise ValueError(f"Unknown field type: {field_type}")
        
        if index is None or index.ntotal == 0:
            logger.warning(f"Index not built for {field_type}")
            return []
        
        if not candidate_ids:
            return []
        
        # Search with larger k to ensure we get all candidates
        search_k = min(k * 10, index.ntotal)
        
        # First do a broad search
        all_results = self.search(query_embedding, field_type, k=search_k)
        
        # Filter to candidate IDs
        candidate_set = set(candidate_ids)
        filtered_results = [(job_id, sim) for job_id, sim in all_results if job_id in candidate_set]
        
        # Return top k
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        return filtered_results[:k]
    
    def save_indices(self, base_path: str):
        """Save all indices to disk."""
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save indices
        faiss.write_index(self.jd_title_index, str(base_path / "jd_title_index.faiss"))
        faiss.write_index(self.jd_skills_index, str(base_path / "jd_skills_index.faiss"))
        faiss.write_index(self.jd_requirement_index, str(base_path / "jd_requirement_index.faiss"))
        
        # Save ID maps
        with open(base_path / "jd_title_id_map.pkl", "wb") as f:
            pickle.dump(self.jd_title_id_map, f)
        with open(base_path / "jd_skills_id_map.pkl", "wb") as f:
            pickle.dump(self.jd_skills_id_map, f)
        with open(base_path / "jd_requirement_id_map.pkl", "wb") as f:
            pickle.dump(self.jd_requirement_id_map, f)
        
        with open(base_path / "jd_title_reverse_map.pkl", "wb") as f:
            pickle.dump(self.jd_title_reverse_map, f)
        with open(base_path / "jd_skills_reverse_map.pkl", "wb") as f:
            pickle.dump(self.jd_skills_reverse_map, f)
        with open(base_path / "jd_requirement_reverse_map.pkl", "wb") as f:
            pickle.dump(self.jd_requirement_reverse_map, f)
        
        logger.info(f"Saved multi-field indices to {base_path}")
    
    def load_indices(self, base_path: str):
        """Load all indices from disk."""
        base_path = Path(base_path)
        
        # Load indices
        self.jd_title_index = faiss.read_index(str(base_path / "jd_title_index.faiss"))
        self.jd_skills_index = faiss.read_index(str(base_path / "jd_skills_index.faiss"))
        self.jd_requirement_index = faiss.read_index(str(base_path / "jd_requirement_index.faiss"))
        
        # Load ID maps
        with open(base_path / "jd_title_id_map.pkl", "rb") as f:
            self.jd_title_id_map = pickle.load(f)
        with open(base_path / "jd_skills_id_map.pkl", "rb") as f:
            self.jd_skills_id_map = pickle.load(f)
        with open(base_path / "jd_requirement_id_map.pkl", "rb") as f:
            self.jd_requirement_id_map = pickle.load(f)
        
        with open(base_path / "jd_title_reverse_map.pkl", "rb") as f:
            self.jd_title_reverse_map = pickle.load(f)
        with open(base_path / "jd_skills_reverse_map.pkl", "rb") as f:
            self.jd_skills_reverse_map = pickle.load(f)
        with open(base_path / "jd_requirement_reverse_map.pkl", "rb") as f:
            self.jd_requirement_reverse_map = pickle.load(f)
        
        logger.info(f"Loaded multi-field indices from {base_path}")
