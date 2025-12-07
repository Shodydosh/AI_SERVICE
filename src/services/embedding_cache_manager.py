"""Smart embedding cache manager with 12-hour refresh cycle."""
import time
import hashlib
import logging
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from threading import Lock
import json

logger = logging.getLogger(__name__)


class EmbeddingCacheManager:
    """
    Smart cache manager for embeddings with 12-hour refresh cycle.
    
    Features:
    - In-memory cache with TTL (12 hours)
    - Content hash tracking to detect changes
    - Thread-safe operations
    - Automatic expiration
    """
    
    def __init__(self, cache_ttl_hours: float = 12.0):
        """
        Initialize cache manager.
        
        Args:
            cache_ttl_hours: Cache TTL in hours (default: 12 hours)
        """
        self.cache_ttl_hours = cache_ttl_hours
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
        logger.info(f"EmbeddingCacheManager initialized with TTL: {cache_ttl_hours} hours")
    
    def _compute_content_hash(self, text: str) -> str:
        """Compute hash of text content to detect changes."""
        if not text:
            return ""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _is_expired(self, cached_item: Dict[str, Any]) -> bool:
        """Check if cached item is expired."""
        if 'timestamp' not in cached_item:
            return True
        
        timestamp = cached_item['timestamp']
        if isinstance(timestamp, datetime):
            age = datetime.now() - timestamp
        else:
            # Handle Unix timestamp
            age = datetime.now() - datetime.fromtimestamp(timestamp)
        
        return age > timedelta(hours=self.cache_ttl_hours)
    
    def get(
        self,
        entity_id: str,
        entity_type: str,
        content_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached embedding.
        
        Args:
            entity_id: Candidate ID or Job ID
            entity_type: 'candidate' or 'job'
            content_hash: Optional content hash to verify content hasn't changed
            
        Returns:
            Cached embedding dict or None if not found/expired
        """
        cache_key = f"{entity_type}:{entity_id}"
        
        with self.lock:
            if cache_key not in self.cache:
                return None
            
            cached_item = self.cache[cache_key]
            
            # Check expiration
            if self._is_expired(cached_item):
                logger.debug(f"Cache expired for {cache_key}")
                del self.cache[cache_key]
                return None
            
            # Check content hash if provided
            if content_hash and cached_item.get('content_hash') != content_hash:
                logger.debug(f"Content changed for {cache_key}, invalidating cache")
                del self.cache[cache_key]
                return None
            
            # Update access time
            cached_item['last_accessed'] = datetime.now()
            return cached_item.get('embedding')
    
    def set(
        self,
        entity_id: str,
        entity_type: str,
        embedding: Any,
        content_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Cache embedding.
        
        Args:
            entity_id: Candidate ID or Job ID
            entity_type: 'candidate' or 'job'
            embedding: Embedding vector or dict
            content_hash: Optional content hash
            metadata: Optional metadata
        """
        cache_key = f"{entity_type}:{entity_id}"
        
        with self.lock:
            self.cache[cache_key] = {
                'embedding': embedding,
                'timestamp': datetime.now(),
                'last_accessed': datetime.now(),
                'content_hash': content_hash,
                'metadata': metadata or {}
            }
            logger.debug(f"Cached embedding for {cache_key}")
    
    def invalidate(self, entity_id: str, entity_type: str):
        """Invalidate cache for specific entity."""
        cache_key = f"{entity_type}:{entity_id}"
        with self.lock:
            if cache_key in self.cache:
                del self.cache[cache_key]
                logger.debug(f"Invalidated cache for {cache_key}")
    
    def clear_expired(self):
        """Clear all expired cache entries."""
        with self.lock:
            expired_keys = [
                key for key, item in self.cache.items()
                if self._is_expired(item)
            ]
            for key in expired_keys:
                del self.cache[key]
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = len(self.cache)
            expired = sum(1 for item in self.cache.values() if self._is_expired(item))
            valid = total - expired
            
            return {
                'total_entries': total,
                'valid_entries': valid,
                'expired_entries': expired,
                'cache_ttl_hours': self.cache_ttl_hours
            }
    
    def needs_refresh(
        self,
        entity_id: str,
        entity_type: str,
        db_timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Check if entity needs embedding refresh.
        
        Args:
            entity_id: Candidate ID or Job ID
            entity_type: 'candidate' or 'job'
            db_timestamp: Timestamp from database (when embedding was last computed)
            
        Returns:
            True if needs refresh, False otherwise
        """
        # Check cache first
        cache_key = f"{entity_type}:{entity_id}"
        with self.lock:
            if cache_key in self.cache:
                if not self._is_expired(self.cache[cache_key]):
                    return False  # Valid cache exists
        
        # Check database timestamp
        if db_timestamp:
            age = datetime.now() - db_timestamp
            if age < timedelta(hours=self.cache_ttl_hours):
                return False  # Database embedding is still fresh
        
        return True  # Needs refresh


# Global cache instance
_global_cache: Optional[EmbeddingCacheManager] = None


def get_cache_manager(ttl_hours: float = 12.0) -> EmbeddingCacheManager:
    """Get or create global cache manager instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = EmbeddingCacheManager(cache_ttl_hours=ttl_hours)
    return _global_cache






