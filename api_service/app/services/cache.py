"""
Caching service for API endpoints.
"""
import json
import asyncio
from typing import Any, Optional, Callable, TypeVar, Dict
from functools import wraps
import hashlib
import time
import redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.core.logging import logger


T = TypeVar('T')


class CacheService:
    """
    Service for caching API responses.
    """
    
    def __init__(self):
        """
        Initialize the cache service.
        """
        # Only create Redis connection if Redis URL is configured
        self._redis_client = None
        if settings.REDIS_URL:
            try:
                self._redis_client = redis.from_url(settings.REDIS_URL)
                logger.info("Redis cache initialized")
            except RedisError as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
        
        # In-memory cache as fallback
        self._local_cache: Dict[str, Dict[str, Any]] = {}
    
    def is_redis_available(self) -> bool:
        """
        Check if Redis is available.
        """
        if not self._redis_client:
            return False
        
        try:
            return self._redis_client.ping()
        except:
            return False
    
    def _create_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Create a cache key from the args and kwargs.
        """
        # Create a string representation of the args and kwargs
        key_parts = [prefix]
        
        if args:
            for arg in args:
                if hasattr(arg, '__dict__'):
                    key_parts.append(str(arg.__dict__))
                else:
                    key_parts.append(str(arg))
        
        if kwargs:
            # Sort kwargs by key to ensure consistent key generation
            for k, v in sorted(kwargs.items()):
                if k == 'db' or k == 'current_user':
                    # Skip DB session and current user
                    continue
                
                if hasattr(v, '__dict__'):
                    key_parts.append(f"{k}:{str(v.__dict__)}")
                else:
                    key_parts.append(f"{k}:{str(v)}")
        
        # Create a hash of the key parts
        key_string = "_".join(key_parts)
        return f"sift_cache:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        """
        # Try to get from Redis first
        if self.is_redis_available():
            try:
                value = self._redis_client.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.error(f"Redis cache get error: {str(e)}")
        
        # Fall back to local cache
        cache_item = self._local_cache.get(key)
        if cache_item:
            if time.time() < cache_item.get('expires_at', 0):
                return cache_item.get('value')
            else:
                # Clean up expired item
                self._local_cache.pop(key, None)
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a value in the cache.
        """
        if ttl is None:
            ttl = settings.CACHE_TTL
        
        # Try to set in Redis first
        if self.is_redis_available():
            try:
                serialized = json.dumps(value)
                return self._redis_client.set(key, serialized, ex=ttl)
            except Exception as e:
                logger.error(f"Redis cache set error: {str(e)}")
        
        # Fall back to local cache
        try:
            self._local_cache[key] = {
                'value': value,
                'expires_at': time.time() + ttl
            }
            return True
        except Exception as e:
            logger.error(f"Local cache set error: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        """
        success = True
        
        # Try to delete from Redis first
        if self.is_redis_available():
            try:
                self._redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis cache delete error: {str(e)}")
                success = False
        
        # Also delete from local cache
        try:
            self._local_cache.pop(key, None)
        except Exception as e:
            logger.error(f"Local cache delete error: {str(e)}")
            success = False
        
        return success
    
    async def flush(self) -> bool:
        """
        Flush all cache entries.
        """
        success = True
        
        # Try to flush Redis first
        if self.is_redis_available():
            try:
                # Only flush keys with the sift_cache prefix
                for key in self._redis_client.keys("sift_cache:*"):
                    self._redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis cache flush error: {str(e)}")
                success = False
        
        # Also flush local cache
        try:
            self._local_cache.clear()
        except Exception as e:
            logger.error(f"Local cache flush error: {str(e)}")
            success = False
        
        return success


# Singleton instance
cache_service = CacheService()


def cache(ttl: int = None, prefix: str = ""):
    """
    Cache decorator for async functions.
    
    Args:
        ttl: Cache TTL in seconds (default: settings.CACHE_TTL)
        prefix: Cache key prefix
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Skip caching if settings.DEBUG is True
            if settings.DEBUG:
                return await func(*args, **kwargs)
            
            # Create cache key
            key_prefix = prefix or func.__name__
            cache_key = cache_service._create_cache_key(key_prefix, *args, **kwargs)
            
            # Try to get from cache
            cached = await cache_service.get(cache_key)
            if cached is not None:
                return cached
            
            # Call the original function
            result = await func(*args, **kwargs)
            
            # Store the result in cache
            await cache_service.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator
