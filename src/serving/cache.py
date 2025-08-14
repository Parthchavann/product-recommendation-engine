import redis
import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union
import numpy as np
from datetime import datetime, timedelta

from ..utils.logger import get_logger


logger = get_logger(__name__)


class RedisCache:
    """Redis-based caching for recommendation system"""
    
    def __init__(self, 
                 host: str = 'localhost',
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 max_connections: int = 10,
                 socket_timeout: int = 5,
                 default_ttl: int = 3600):
        
        self.default_ttl = default_ttl
        
        # Create connection pool
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            decode_responses=False  # We'll handle encoding manually
        )
        
        # Create Redis client
        self.redis_client = redis.Redis(connection_pool=self.pool)
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError:
            logger.warning(f"Could not connect to Redis at {host}:{port}")
            self.redis_client = None
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate a cache key from parameters"""
        
        # Create a deterministic key from parameters
        key_parts = [prefix]
        
        for key, value in sorted(kwargs.items()):
            if isinstance(value, (list, dict, np.ndarray)):
                # Hash complex objects
                value_str = str(value)
                value_hash = hashlib.md5(value_str.encode()).hexdigest()[:8]
                key_parts.append(f"{key}:{value_hash}")
            else:
                key_parts.append(f"{key}:{value}")
        
        return ":".join(key_parts)
    
    def set(self, 
            key: str, 
            value: Any, 
            ttl: Optional[int] = None,
            serialization: str = 'json') -> bool:
        """
        Set a value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialization: Serialization method ('json' or 'pickle')
            
        Returns:
            Success status
        """
        
        if self.redis_client is None:
            return False
        
        try:
            # Serialize value
            if serialization == 'json':
                serialized_value = json.dumps(value, default=self._json_serializer)
            elif serialization == 'pickle':
                serialized_value = pickle.dumps(value)
            else:
                raise ValueError(f"Unsupported serialization: {serialization}")
            
            # Set in Redis
            ttl = ttl or self.default_ttl
            result = self.redis_client.setex(key, ttl, serialized_value)
            
            logger.debug(f"Cached key: {key}, TTL: {ttl}s")
            return result
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def get(self, 
            key: str, 
            serialization: str = 'json') -> Optional[Any]:
        """
        Get a value from cache
        
        Args:
            key: Cache key
            serialization: Serialization method used
            
        Returns:
            Cached value or None if not found
        """
        
        if self.redis_client is None:
            return None
        
        try:
            serialized_value = self.redis_client.get(key)
            
            if serialized_value is None:
                return None
            
            # Deserialize value
            if serialization == 'json':
                value = json.loads(serialized_value)
            elif serialization == 'pickle':
                value = pickle.loads(serialized_value)
            else:
                raise ValueError(f"Unsupported serialization: {serialization}")
            
            logger.debug(f"Cache hit for key: {key}")
            return value
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        
        if self.redis_client is None:
            return False
        
        try:
            result = self.redis_client.delete(key)
            logger.debug(f"Deleted cache key: {key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        
        if self.redis_client is None:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def set_recommendations(self,
                           user_id: int,
                           recommendations: List[Dict],
                           ttl: Optional[int] = None,
                           **cache_params) -> bool:
        """Cache user recommendations"""
        
        cache_key = self._generate_key("rec", user_id=user_id, **cache_params)
        
        cache_data = {
            'recommendations': recommendations,
            'cached_at': datetime.now().isoformat(),
            'user_id': user_id,
            'cache_params': cache_params
        }
        
        return self.set(cache_key, cache_data, ttl=ttl)
    
    def get_recommendations(self,
                           user_id: int,
                           **cache_params) -> Optional[Dict]:
        """Get cached user recommendations"""
        
        cache_key = self._generate_key("rec", user_id=user_id, **cache_params)
        return self.get(cache_key)
    
    def set_user_embedding(self,
                          user_id: int,
                          embedding: np.ndarray,
                          ttl: Optional[int] = None) -> bool:
        """Cache user embedding"""
        
        cache_key = self._generate_key("user_emb", user_id=user_id)
        
        # Convert numpy array to list for JSON serialization
        embedding_data = {
            'embedding': embedding.tolist(),
            'shape': embedding.shape,
            'user_id': user_id,
            'cached_at': datetime.now().isoformat()
        }
        
        return self.set(cache_key, embedding_data, ttl=ttl)
    
    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get cached user embedding"""
        
        cache_key = self._generate_key("user_emb", user_id=user_id)
        data = self.get(cache_key)
        
        if data and 'embedding' in data:
            return np.array(data['embedding'])
        
        return None
    
    def set_item_embeddings(self,
                           item_embeddings: Dict[int, np.ndarray],
                           ttl: Optional[int] = None) -> bool:
        """Cache item embeddings (batch)"""
        
        cache_key = "item_embeddings:all"
        
        # Convert numpy arrays to lists
        embeddings_data = {}
        for item_id, embedding in item_embeddings.items():
            embeddings_data[str(item_id)] = {
                'embedding': embedding.tolist(),
                'shape': embedding.shape
            }
        
        cache_data = {
            'embeddings': embeddings_data,
            'cached_at': datetime.now().isoformat(),
            'count': len(item_embeddings)
        }
        
        return self.set(cache_key, cache_data, ttl=ttl, serialization='pickle')
    
    def get_item_embeddings(self) -> Optional[Dict[int, np.ndarray]]:
        """Get cached item embeddings"""
        
        cache_key = "item_embeddings:all"
        data = self.get(cache_key, serialization='pickle')
        
        if data and 'embeddings' in data:
            embeddings = {}
            for item_id_str, emb_data in data['embeddings'].items():
                item_id = int(item_id_str)
                embeddings[item_id] = np.array(emb_data['embedding'])
            return embeddings
        
        return None
    
    def set_candidates(self,
                      user_id: int,
                      candidates: List[Dict],
                      retrieval_params: Dict,
                      ttl: Optional[int] = None) -> bool:
        """Cache candidate items for a user"""
        
        cache_key = self._generate_key("candidates", user_id=user_id, **retrieval_params)
        
        cache_data = {
            'candidates': candidates,
            'cached_at': datetime.now().isoformat(),
            'user_id': user_id,
            'retrieval_params': retrieval_params
        }
        
        return self.set(cache_key, cache_data, ttl=ttl)
    
    def get_candidates(self,
                      user_id: int,
                      retrieval_params: Dict) -> Optional[List[Dict]]:
        """Get cached candidates for a user"""
        
        cache_key = self._generate_key("candidates", user_id=user_id, **retrieval_params)
        data = self.get(cache_key)
        
        if data and 'candidates' in data:
            return data['candidates']
        
        return None
    
    def invalidate_user_cache(self, user_id: int) -> int:
        """Invalidate all cache entries for a user"""
        
        if self.redis_client is None:
            return 0
        
        try:
            # Find all keys related to this user
            patterns = [
                f"rec:user_id:{user_id}:*",
                f"user_emb:user_id:{user_id}",
                f"candidates:user_id:{user_id}:*"
            ]
            
            deleted_count = 0
            
            for pattern in patterns:
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted_count += self.redis_client.delete(*keys)
            
            logger.info(f"Invalidated {deleted_count} cache entries for user {user_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error invalidating user cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        if self.redis_client is None:
            return {'status': 'disconnected'}
        
        try:
            info = self.redis_client.info()
            
            return {
                'status': 'connected',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'used_memory': info.get('used_memory', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(
                    info.get('keyspace_hits', 0),
                    info.get('keyspace_misses', 0)
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def flush_all(self) -> bool:
        """Flush all cache entries (use with caution!)"""
        
        if self.redis_client is None:
            return False
        
        try:
            result = self.redis_client.flushdb()
            logger.warning("Flushed all cache entries")
            return result
            
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for complex objects"""
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return str(obj)
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate"""
        
        total = hits + misses
        if total == 0:
            return 0.0
        
        return hits / total
    
    def close(self):
        """Close Redis connections"""
        
        if self.pool:
            self.pool.disconnect()
            logger.info("Disconnected from Redis")


class InMemoryCache:
    """In-memory cache fallback for when Redis is not available"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = {}
        self._access_times = {}
        
        logger.info(f"Initialized in-memory cache with max_size={max_size}")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache"""
        
        try:
            # Remove expired entries
            self._cleanup_expired()
            
            # Remove oldest entries if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()
            
            # Set value with expiration
            expire_time = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
            self._cache[key] = (value, expire_time)
            self._access_times[key] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting in-memory cache key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        
        try:
            if key not in self._cache:
                return None
            
            value, expire_time = self._cache[key]
            
            # Check if expired
            if datetime.now() > expire_time:
                del self._cache[key]
                del self._access_times[key]
                return None
            
            # Update access time
            self._access_times[key] = datetime.now()
            
            return value
            
        except Exception as e:
            logger.error(f"Error getting in-memory cache key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        
        try:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting in-memory cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self._cache
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        
        now = datetime.now()
        expired_keys = []
        
        for key, (value, expire_time) in self._cache.items():
            if now > expire_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            del self._access_times[key]
    
    def _evict_oldest(self):
        """Evict oldest accessed entry"""
        
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        self._cleanup_expired()
        
        return {
            'type': 'in_memory',
            'size': len(self._cache),
            'max_size': self.max_size,
            'utilization': len(self._cache) / self.max_size
        }


class CacheManager:
    """High-level cache manager that handles fallback to in-memory cache"""
    
    def __init__(self, 
                 redis_config: Optional[Dict] = None,
                 fallback_to_memory: bool = True,
                 memory_cache_size: int = 1000):
        
        # Try to initialize Redis cache
        self.redis_cache = None
        if redis_config:
            try:
                self.redis_cache = RedisCache(**redis_config)
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
        
        # Initialize in-memory fallback
        self.memory_cache = None
        if fallback_to_memory:
            self.memory_cache = InMemoryCache(max_size=memory_cache_size)
        
        # Determine active cache
        self.active_cache = self.redis_cache or self.memory_cache
        
        if self.active_cache == self.redis_cache:
            logger.info("Using Redis cache")
        elif self.active_cache == self.memory_cache:
            logger.info("Using in-memory cache")
        else:
            logger.warning("No cache available")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if self.active_cache:
            return self.active_cache.set(key, value, ttl)
        return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if self.active_cache:
            return self.active_cache.get(key)
        return None
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if self.active_cache:
            return self.active_cache.delete(key)
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        if self.active_cache:
            return self.active_cache.exists(key)
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.active_cache:
            return self.active_cache.get_cache_stats()
        return {'status': 'no_cache'}