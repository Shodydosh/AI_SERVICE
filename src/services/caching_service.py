"""Caching Service với Redis cho frequent queries."""
from typing import List, Dict, Optional, Any
import logging
import hashlib
import json
import pickle

logger = logging.getLogger(__name__)


class CachingService:
    """
    Caching Service sử dụng Redis (hoặc in-memory fallback) để cache:
    1. Candidate recommendations
    2. FAISS search results
    3. Embedding computations
    """
    
    def __init__(
        self,
        use_redis: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        default_ttl: int = 3600
    ):
        """
        Initialize caching service.
        
        Args:
            use_redis: Whether to use Redis (if False, use in-memory cache)
            redis_host: Redis host
            redis_port: Redis port
            redis_db: Redis database number
            default_ttl: Default TTL in seconds (1 hour)
        """
        self.use_redis = use_redis
        self.default_ttl = default_ttl
        self.redis_client = None
        self.memory_cache = {}  # Fallback in-memory cache
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False  # Use bytes for pickle
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"CachingService initialized with Redis ({redis_host}:{redis_port})")
            except ImportError:
                logger.warning("redis not installed, falling back to in-memory cache")
                self.use_redis = False
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}, falling back to in-memory cache")
                self.use_redis = False
        
        if not self.use_redis:
            logger.info("CachingService initialized with in-memory cache")
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key từ arguments.
        
        Args:
            prefix: Key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Create hash from args and kwargs
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"{prefix}:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if self.use_redis and self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
            except Exception as e:
                logger.warning(f"Error getting from Redis cache: {e}")
        
        # Fallback to memory cache
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: self.default_ttl)
            
        Returns:
            True if successful
        """
        if ttl is None:
            ttl = self.default_ttl
        
        try:
            serialized = pickle.dumps(value)
            
            if self.use_redis and self.redis_client:
                try:
                    self.redis_client.setex(key, ttl, serialized)
                    return True
                except Exception as e:
                    logger.warning(f"Error setting Redis cache: {e}")
            
            # Fallback to memory cache (no TTL for simplicity)
            self.memory_cache[key] = value
            return True
        except Exception as e:
            logger.error(f"Error serializing cache value: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Error deleting from Redis cache: {e}")
        
        # Also delete from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        return True
    
    def cache_candidate_recommendations(
        self,
        candidate_id: str,
        recommendations: List[Dict],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache candidate recommendations.
        
        Args:
            candidate_id: Candidate ID
            recommendations: List of recommendations
            ttl: Time to live (default: 1 hour)
            
        Returns:
            True if successful
        """
        key = f"candidate_recommendations:{candidate_id}"
        return self.set(key, recommendations, ttl)
    
    def get_cached_recommendations(
        self,
        candidate_id: str
    ) -> Optional[List[Dict]]:
        """
        Get cached candidate recommendations.
        
        Args:
            candidate_id: Candidate ID
            
        Returns:
            Cached recommendations or None
        """
        key = f"candidate_recommendations:{candidate_id}"
        return self.get(key)
    
    def cache_faiss_search(
        self,
        query_vector_hash: str,
        results: List[Dict],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache FAISS search results.
        
        Args:
            query_vector_hash: Hash of query vector
            results: Search results
            ttl: Time to live (default: 30 minutes)
            
        Returns:
            True if successful
        """
        if ttl is None:
            ttl = 1800  # 30 minutes for FAISS results
        
        key = f"faiss_search:{query_vector_hash}"
        return self.set(key, results, ttl)
    
    def get_cached_faiss_search(
        self,
        query_vector_hash: str
    ) -> Optional[List[Dict]]:
        """
        Get cached FAISS search results.
        
        Args:
            query_vector_hash: Hash of query vector
            
        Returns:
            Cached results or None
        """
        key = f"faiss_search:{query_vector_hash}"
        return self.get(key)
    
    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            pattern: Pattern to match (if None, clear all)
            
        Returns:
            Number of keys deleted
        """
        count = 0
        
        if self.use_redis and self.redis_client:
            try:
                if pattern:
                    keys = self.redis_client.keys(pattern)
                    if keys:
                        count = self.redis_client.delete(*keys)
                else:
                    # Clear all (use with caution!)
                    self.redis_client.flushdb()
                    count = -1  # Unknown count
            except Exception as e:
                logger.warning(f"Error clearing Redis cache: {e}")
        
        # Also clear memory cache
        if pattern:
            keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
            for k in keys_to_delete:
                del self.memory_cache[k]
                count += 1
        else:
            self.memory_cache.clear()
            count = -1
        
        return count

