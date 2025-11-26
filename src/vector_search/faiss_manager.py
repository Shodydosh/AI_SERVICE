"""FAISS index manager for efficient vector similarity search."""
import faiss
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from sqlalchemy.orm import Session
import logging
from pathlib import Path
import pickle

from src.database.models import JobDescriptionEmbedding, CandidateEmbedding

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """Manages FAISS indices for fast vector similarity search."""
    
    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "Flat",
        index_params: Optional[Dict[str, Any]] = None,
        normalize: bool = True
    ):
        """
        Initialize FAISS index manager.
        
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
        
        self.jd_index: Optional[faiss.Index] = None
        self.candidate_index: Optional[faiss.Index] = None
        
        self.jd_id_map: Dict[int, str] = {}  # FAISS index -> job_id
        self.candidate_id_map: Dict[int, str] = {}  # FAISS index -> candidate_id
        
        self._initialize_indexes()
    
    def _initialize_indexes(self):
        """Initialize FAISS indexes based on type."""
        if self.index_type == "Flat":
            # Exact search using L2 distance
            self.jd_index = faiss.IndexFlatL2(self.dimension)
            self.candidate_index = faiss.IndexFlatL2(self.dimension)
        
        elif self.index_type == "IVF":
            # Inverted file index for faster approximate search
            nlist = self.index_params.get('nlist', 100)  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.jd_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.candidate_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World for very fast approximate search
            M = self.index_params.get('M', 32)  # Number of connections
            self.jd_index = faiss.IndexHNSWFlat(self.dimension, M)
            self.candidate_index = faiss.IndexHNSWFlat(self.dimension, M)
            
            # Set ef_construction if provided (higher = better quality index)
            if 'ef_construction' in self.index_params:
                self.jd_index.hnsw.efConstruction = self.index_params['ef_construction']
                self.candidate_index.hnsw.efConstruction = self.index_params['ef_construction']
            else:
                # Default to higher ef_construction for better index quality
                self.jd_index.hnsw.efConstruction = 200
                self.candidate_index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return vectors / norms
        return vectors
    
    def build_index_from_db(
        self,
        db: Session,
        dataset_type: str = "jd",
        batch_size: int = 1000
    ):
        """
        Build FAISS index from PostgreSQL database.
        
        Args:
            db: Database session
            dataset_type: "jd" or "candidate"
            batch_size: Batch size for processing
        """
        logger.info(f"Building FAISS index for {dataset_type} dataset...")
        
        if dataset_type == "jd":
            embeddings_query = db.query(JobDescriptionEmbedding).all()
            index = self.jd_index
            id_map = self.jd_id_map
        elif dataset_type == "candidate":
            embeddings_query = db.query(CandidateEmbedding).all()
            index = self.candidate_index
            id_map = self.candidate_id_map
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        if not embeddings_query:
            logger.warning(f"No embeddings found for {dataset_type} dataset")
            return
        
        # Extract embeddings and IDs
        vectors = []
        ids = []
        
        from tqdm import tqdm
        for emb_obj in tqdm(embeddings_query, desc=f"Loading {dataset_type} embeddings", unit="record"):
            if dataset_type == "jd":
                vectors.append(emb_obj.embedding)
                ids.append(emb_obj.job_id)
            else:
                vectors.append(emb_obj.embedding)
                ids.append(emb_obj.candidate_id)
        
        # Convert to numpy array
        vectors = np.array(vectors, dtype=np.float32)
        
        # Normalize if needed
        vectors = self._normalize_vectors(vectors)
        
        # Build index
        if self.index_type == "IVF":
            # Train index first
            logger.info("Training IVF index...")
            from tqdm import tqdm
            with tqdm(total=1, desc="Training IVF index") as pbar:
                index.train(vectors)
                pbar.update(1)
        
        # Add vectors to index
        logger.info(f"Adding {len(vectors)} vectors to index...")
        from tqdm import tqdm
        with tqdm(total=1, desc="Building FAISS index") as pbar:
            index.add(vectors)
            pbar.update(1)
        
        # Build ID mapping
        for idx, entity_id in enumerate(ids):
            id_map[idx] = entity_id
        
        logger.info(f"✓ FAISS index built successfully with {len(vectors)} vectors")
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        dataset_type: str = "jd"
    ) -> List[Tuple[str, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            dataset_type: "jd" or "candidate"
        
        Returns:
            List of (entity_id, distance) tuples
        """
        if dataset_type == "jd":
            index = self.jd_index
            id_map = self.jd_id_map
        elif dataset_type == "candidate":
            index = self.candidate_index
            id_map = self.candidate_id_map
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        if index is None or index.ntotal == 0:
            logger.warning(f"Index not built for {dataset_type}")
            return []
        
        # Convert query to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Normalize if needed
        query_vector = self._normalize_vectors(query_vector)
        
        # Set search parameters for IVF and HNSW
        if self.index_type == "IVF":
            index.nprobe = self.index_params.get('nprobe', 10)
        elif self.index_type == "HNSW":
            # Dynamically adjust ef_search based on k for better precision
            base_ef_search = self.index_params.get('ef_search', 32)
            # For small k (like 15), use higher ef_search for better precision
            # For larger k, increase proportionally but cap at reasonable value
            if k <= 20:
                # For small k, use higher ef_search for maximum precision
                ef_search = max(base_ef_search * 2, k * 4, 64)
            else:
                # For larger k, scale proportionally
                ef_search = min(max(k * 2, base_ef_search), 500)
            index.hnsw.efSearch = ef_search
        
        # Search
        distances, indices = index.search(query_vector, k)
        
        # Map indices to entity IDs
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(id_map) and idx >= 0:
                entity_id = id_map[idx]
                # Convert L2 distance to similarity score (for cosine: 1 - distance/2)
                if self.normalize:
                    similarity = 1 - (dist / 2.0)  # Cosine similarity from L2 distance
                else:
                    similarity = 1 / (1 + dist)  # Convert distance to similarity
                results.append((entity_id, float(similarity)))
        
        return results
    
    def save_index(self, filepath: str, dataset_type: str = "jd"):
        """Save FAISS index to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if dataset_type == "jd":
            index = self.jd_index
            id_map = self.jd_id_map
        else:
            index = self.candidate_index
            id_map = self.candidate_id_map
        
        if index is None:
            logger.warning(f"No index to save for {dataset_type}")
            return
        
        # Save FAISS index
        faiss.write_index(index, str(path))
        
        # Save ID mapping
        id_map_path = path.with_suffix('.pkl')
        with open(id_map_path, 'wb') as f:
            pickle.dump(id_map, f)
        
        logger.info(f"Saved {dataset_type} index to {filepath}")
    
    def load_index(self, filepath: str, dataset_type: str = "jd"):
        """Load FAISS index from disk."""
        path = Path(filepath)
        
        if not path.exists():
            logger.warning(f"Index file not found: {filepath}")
            return
        
        # Load FAISS index
        if dataset_type == "jd":
            self.jd_index = faiss.read_index(str(path))
        else:
            self.candidate_index = faiss.read_index(str(path))
        
        # Load ID mapping
        id_map_path = path.with_suffix('.pkl')
        if id_map_path.exists():
            with open(id_map_path, 'rb') as f:
                if dataset_type == "jd":
                    self.jd_id_map = pickle.load(f)
                else:
                    self.candidate_id_map = pickle.load(f)
        
        logger.info(f"Loaded {dataset_type} index from {filepath}")
    
    def get_index_stats(self, dataset_type: str = "jd") -> Dict[str, Any]:
        """Get statistics about the index."""
        if dataset_type == "jd":
            index = self.jd_index
            id_map = self.jd_id_map
        else:
            index = self.candidate_index
            id_map = self.candidate_id_map
        
        if index is None:
            return {"status": "not_built"}
        
        stats = {
            "status": "built",
            "total_vectors": index.ntotal,
            "dimension": index.d,
            "index_type": self.index_type,
            "id_map_size": len(id_map),
            "is_trained": index.is_trained if hasattr(index, 'is_trained') else True
        }
        
        if self.index_type == "IVF":
            stats["nlist"] = index.nlist
            stats["nprobe"] = index.nprobe
        
        if self.index_type == "HNSW":
            stats["M"] = index.hnsw.M
            stats["ef_construction"] = index.hnsw.efConstruction
            stats["ef_search"] = index.hnsw.efSearch
        
        return stats
    
    def rebuild_index(
        self,
        db: Session,
        dataset_type: str = "jd",
        save_path: Optional[str] = None
    ):
        """
        Rebuild FAISS index from database.
        
        Args:
            db: Database session
            dataset_type: "jd" or "candidate"
            save_path: Optional path to save index after rebuilding
        """
        logger.info(f"Rebuilding FAISS index for {dataset_type}...")
        
        # Re-initialize index
        self._initialize_indexes()
        
        # Build from database
        self.build_index_from_db(db, dataset_type=dataset_type)
        
        # Save if path provided
        if save_path:
            self.save_index(save_path, dataset_type=dataset_type)
        
        logger.info(f"✓ Index rebuilt successfully")
    
    def add_vector(
        self,
        vector: List[float],
        entity_id: str,
        dataset_type: str = "jd"
    ):
        """
        Add a single vector to the index.
        
        Args:
            vector: Embedding vector
            entity_id: ID of the entity
            dataset_type: "jd" or "candidate"
        """
        if dataset_type == "jd":
            index = self.jd_index
            id_map = self.jd_id_map
        else:
            index = self.candidate_index
            id_map = self.candidate_id_map
        
        if index is None:
            raise ValueError(f"Index not initialized for {dataset_type}")
        
        # Convert to numpy array and normalize
        vector_array = np.array([vector], dtype=np.float32)
        vector_array = self._normalize_vectors(vector_array)
        
        # Add to index
        current_size = index.ntotal
        index.add(vector_array)
        
        # Update ID map
        id_map[current_size] = entity_id
        
        logger.info(f"Added vector for {entity_id} to {dataset_type} index")
    
    def remove_vector(
        self,
        entity_id: str,
        dataset_type: str = "jd"
    ) -> bool:
        """
        Remove a vector from the index by entity ID.
        Note: FAISS doesn't support direct removal, so this requires rebuilding.
        
        Args:
            entity_id: ID of the entity to remove
            dataset_type: "jd" or "candidate"
        
        Returns:
            True if removed, False if not found
        """
        if dataset_type == "jd":
            id_map = self.jd_id_map
        else:
            id_map = self.candidate_id_map
        
        # Find index position
        index_pos = None
        for idx, eid in id_map.items():
            if eid == entity_id:
                index_pos = idx
                break
        
        if index_pos is None:
            return False
        
        # FAISS doesn't support direct removal, so we need to rebuild
        logger.warning(f"FAISS doesn't support direct removal. Rebuild required to remove {entity_id}")
        return False
    
    def update_vector(
        self,
        vector: List[float],
        entity_id: str,
        dataset_type: str = "jd"
    ):
        """
        Update a vector in the index.
        Note: FAISS doesn't support direct updates, so this removes and re-adds.
        
        Args:
            vector: New embedding vector
            entity_id: ID of the entity
            dataset_type: "jd" or "candidate"
        """
        # Remove old vector (requires rebuild)
        removed = self.remove_vector(entity_id, dataset_type)
        if not removed:
            logger.warning(f"Could not remove old vector for {entity_id}, adding as new")
        
        # Add new vector
        self.add_vector(vector, entity_id, dataset_type)

