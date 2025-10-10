# cache_utils.py
"""
Simple in-memory caching for sheet data with TTL.
Prevents repeated downloads of the same Google Sheets.
"""

import hashlib
import time
from typing import Any, Callable, Dict, Optional
from functools import wraps

import config
import logging_utils as log_util

logger = log_util.get_logger("cache")


class CacheEntry:
    """Single cache entry with TTL."""
    
    def __init__(self, value: Any, ttl_seconds: int = config.CACHE_TTL_SECONDS):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl_seconds
    
    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        age = time.time() - self.created_at
        return age > self.ttl
    
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at


class SimpleCache:
    """Thread-safe in-memory cache with TTL."""
    
    def __init__(self, max_size: int = config.CACHE_MAX_SIZE):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
    
    def _make_key(self, *args, **kwargs) -> str:
        """Create a cache key from arguments."""
        key_str = str((args, sorted(kwargs.items())))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value, or None if expired/missing."""
        entry = self.cache.get(key)
        if entry is None:
            logger.debug(f"Cache miss: {key}")
            return None
        
        if entry.is_expired():
            logger.debug(f"Cache expired: {key} (age: {entry.age_seconds():.1f}s)")
            del self.cache[key]
            return None
        
        logger.debug(f"Cache hit: {key} (age: {entry.age_seconds():.1f}s)")
        return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Store a value."""
        if len(self.cache) >= self.max_size:
            # Evict oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
            logger.debug(f"Cache eviction: {oldest_key} (full, size={self.max_size})")
            del self.cache[oldest_key]
        
        ttl = ttl_seconds or config.CACHE_TTL_SECONDS
        self.cache[key] = CacheEntry(value, ttl)
        logger.debug(f"Cache set: {key} (ttl: {ttl}s)")
    
    def clear(self):
        """Clear all cache entries."""
        logger.info(f"Cache cleared ({len(self.cache)} entries removed)")
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "entries": [
                {
                    "key": k,
                    "age_seconds": v.age_seconds(),
                    "ttl_seconds": v.ttl,
                    "expired": v.is_expired()
                }
                for k, v in self.cache.items()
            ]
        }


# Global cache instance
_cache = SimpleCache()


def cache_get(key: str) -> Optional[Any]:
    """Retrieve from global cache."""
    return _cache.get(key)


def cache_set(key: str, value: Any, ttl_seconds: Optional[int] = None):
    """Store in global cache."""
    _cache.set(key, value, ttl_seconds)


def cache_clear():
    """Clear global cache."""
    _cache.clear()


def cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _cache.stats()


def cached_url_request(func: Callable) -> Callable:
    """
    Decorator to cache HTTP requests by URL.
    Useful for fetch_xlsx and similar functions.
    
    The decorated function should accept a URL as first argument.
    """
    @wraps(func)
    def wrapper(url: str, *args, **kwargs):
        cache_key = _cache._make_key(func.__name__, url, *args)
        
        # Try cache first
        cached = cache_get(cache_key)
        if cached is not None:
            logger.info(f"Cached request for {func.__name__}", extra={"url": url})
            return cached
        
        # Cache miss: execute function
        logger.info(f"Fetching {func.__name__}", extra={"url": url})
        result = func(url, *args, **kwargs)
        
        # Store in cache
        cache_set(cache_key, result)
        return result
    
    return wrapper
