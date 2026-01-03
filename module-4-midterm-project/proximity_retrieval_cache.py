"""
Proximity Retrieval Cache Module

This module implements a multi-modal proximity-based document retrieval cache system that supports
two main modes of operation:
1. Proximity-FLAT: Linear scan over all cached entries (simplest, most accurate, but O(n))
2. Proximity-LSH: Locality-Sensitive Hashing with random hyperplanes for approximate nearest neighbor search
3. RedisSearch: Uses Redis' built-in vector search capabilities (when available)

The cache is designed to store and retrieve document contexts based on semantic similarity of queries,
reducing the need to hit the vector database for similar queries.

Key concepts:
- τ (tolerance): Maximum cosine distance for a cache hit (lower = more strict matching)
- ρ (rerank_factor): Multiplier for over-fetching documents before re-ranking
- LRU eviction: Removes least recently used entries when capacity is reached
"""

from typing import Dict, List, Optional, Tuple, Union
import base64
import hashlib
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import redis
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config
import config

# Initialize the embeddings model
_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

try:
    # Try to import Redis Search modules - these are only available with redis-py >= 4.0.0
    # and when the Redis server has the RediSearch module enabled
    from redis.commands.search.field import TextField, NumericField, VectorField
    from redis.commands.search.query import Query
    try:
        # Handle different import paths for different redis-py versions
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    except ImportError:
        from redis.commands.search.index_definition import IndexDefinition, IndexType
    _REDISEARCH_CLIENT_AVAILABLE = True
except (ImportError, AttributeError) as e:
    # Gracefully degrade functionality if Redis Search is not available
    TextField = NumericField = VectorField = None
    IndexDefinition = IndexType = None
    Query = None
    _REDISEARCH_CLIENT_AVAILABLE = False



def _normalize_query(q: str) -> str:
    """Normalize a query string for consistent caching.
    
    Args:
        q: Input query string
        
    Returns:
        Normalized query string with:
        - Leading/trailing whitespace removed
        - Converted to lowercase
        - Multiple spaces collapsed to single spaces
    """
    return " ".join((q or "").strip().lower().split())


def _stable_key(q: str) -> str:
    """Generate a stable cache key for a query string.
    
    The key is deterministic and will be the same for identical queries.
    
    Args:
        q: Input query string
        
    Returns:
        A cache key string with format: {PREFIX}{SHA256(normalized_query)}
    """
    nq = _normalize_query(q)
    digest = hashlib.sha256(nq.encode("utf-8")).hexdigest()
    return f"{config.PROXIMITY_CACHE_KEY_PREFIX}{digest}"


def _to_float32_bytes(vec: List[float]) -> bytes:
    """Convert a list of floats to a packed bytes object.
    
    Args:
        vec: List of 32-bit floating point numbers
        
    Returns:
        Packed bytes representation of the input vector
    """
    return struct.pack(f"<{len(vec)}f", *vec)


def _from_float32_bytes(b: bytes) -> List[float]:
    """Convert a packed bytes object back to a list of floats.
    
    Args:
        b: Packed bytes (from _to_float32_bytes)
        
    Returns:
        List of 32-bit floating point numbers
    """
    n = len(b) // 4  # Each float32 is 4 bytes
    return list(struct.unpack(f"<{n}f", b))


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """Calculate the cosine distance between two vectors.
    
    The cosine distance is defined as 1 - cosine_similarity(a, b).
    
    Args:
        a: First vector
        b: Second vector (must be same length as a)
        
    Returns:
        Cosine distance between a and b, in range [0.0, 2.0]
        - 0.0 means identical vectors
        - 1.0 means orthogonal vectors
        - 2.0 means opposite vectors
    """
    dot = 0.0  # Dot product of a and b
    na = 0.0   # Squared L2 norm of a
    nb = 0.0   # Squared L2 norm of b
    
    # Calculate dot product and squared norms in a single pass
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    
    # Handle zero-vector edge cases
    if na <= 0.0 or nb <= 0.0:
        return 1.0  # Maximum distance if either vector is zero
        
    # Calculate cosine similarity and convert to distance
    cosine_sim = dot / (math.sqrt(na) * math.sqrt(nb))
    return 1.0 - cosine_sim


def _dot(a: List[float], b: List[float]) -> float:
    """Calculate the dot product of two vectors.
    
    Args:
        a: First vector
        b: Second vector (must be same length as a)
        
    Returns:
        Dot product of a and b
    """
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


class ProximityRetrievalCache:
    """
    A proximity-based retrieval cache that stores and retrieves document contexts based on
    semantic similarity of queries. Supports three modes of operation:
    - REDISEARCH: Uses Redis' built-in vector search (fastest, requires Redis Stack)
    - LSH: Locality-Sensitive Hashing with random hyperplanes (fast, approximate)
    - FLAT: Linear scan over all entries (slowest, most accurate)
    
    The cache implements LRU eviction and supports configurable capacity, similarity threshold,
    and re-ranking of results.
    """
    
    def __init__(
        self,
        redis_url: str = config.REDIS_URL,
        index_name: str = config.PROXIMITY_CACHE_INDEX_NAME,
        ttl_seconds: int = config.CACHE_TTL_SECONDS,
        capacity: int = config.PROXIMITY_CACHE_CAPACITY,
        tolerance: float = config.PROXIMITY_DISTANCE_TOLERANCE,
        top_k: int = config.PROXIMITY_RETRIEVAL_TOP_K,
        rerank_factor: int = config.PROXIMITY_RERANK_FACTOR,
        eviction_policy: str = config.PROXIMITY_EVICTION_POLICY,
    ):
        """Initialize the ProximityRetrievalCache.
        
        Args:
            redis_url: URL for connecting to Redis
            index_name: Name of the Redis Search index (for REDISEARCH mode)
            ttl_seconds: Time-to-live for cached entries in seconds
            capacity: Maximum number of entries to store before evicting
            tolerance: Maximum cosine distance (τ) for considering a cache hit
            top_k: Number of documents to return for each query
            rerank_factor: Factor for over-fetching documents (ρ) before re-ranking
            eviction_policy: Cache eviction policy ('LRU' or 'FIFO')
        """
        # Basic configuration
        self.redis_url = redis_url
        self.index_name = index_name
        self.ttl_seconds = ttl_seconds
        self.capacity = capacity
        self.tolerance = tolerance  # τ in the paper
        self.top_k = top_k         # Number of docs to return
        self.rerank_factor = rerank_factor  # ρ in the paper (reranking factor)
        self.eviction_policy = (eviction_policy or "LRU").upper()

        # LSH configuration (for Proximity-LSH mode)
        self.mode = str(getattr(config, "PROXIMITY_CACHE_MODE", "LSH")).upper()
        self.lsh_num_bits = int(getattr(config, "PROXIMITY_LSH_NUM_BITS", 16))  # Number of hyperplanes
        self.lsh_bucket_capacity = int(getattr(config, "PROXIMITY_LSH_BUCKET_CAPACITY", 20))  # Max entries per bucket
        self.lsh_seed = int(getattr(config, "PROXIMITY_LSH_SEED", 42))  # For deterministic hyperplane generation

        # Internal state
        self._dim = None  # Will be set to the embedding dimension on first use
        self._disabled_reason_logged = False  # Track if we've logged why the cache is disabled
        self._index_ready_logged = False  # Track if we've logged index creation

        # LSH hyperplanes (lazily initialized)
        self._lsh_hyperplanes: Optional[List[List[float]]] = None

        # Redis key for the global LRU sorted set (used in FLAT and REDISEARCH modes)
        self._lru_zset_key = f"{config.PROXIMITY_CACHE_KEY_PREFIX}__lru__"

        # Initialize Redis client and enabled status
        self.enabled = True
        try:
            self.redis = redis.Redis.from_url(redis_url)
            self.redis.ping()  # Test connection
        except Exception as e:
            logging.error(f"Proximity retrieval cache Redis connection error: {e}")
            self.redis = None
            self.enabled = False

    def _log_disabled_reason(self) -> None:
        """Log the reason why the cache is disabled (if it is).
        
        Only logs the reason once per instance to avoid log spam.
        """
        if self._disabled_reason_logged:
            return
            
        self._disabled_reason_logged = True

        if not bool(getattr(config, "PROXIMITY_CACHE_ENABLED", True)):
            logging.info("Proximity cache disabled by config (PROXIMITY_CACHE_ENABLED=False)")
            return
            
        if self.redis is None:
            logging.info("Proximity cache disabled (Redis connection not available)")
            return
            
        if self.mode == "REDISEARCH" and not _REDISEARCH_CLIENT_AVAILABLE:
            logging.info(
                "Proximity cache disabled (redis-py RediSearch client not available). "
                "Upgrade dependency: pip install -U 'redis>=5.0.0'"
            )
            return

    def is_available(self) -> bool:
        """Check if the cache is available for use.
        
        Returns:
            bool: True if the cache is enabled and ready to use, False otherwise
        """
        if not bool(getattr(config, "PROXIMITY_CACHE_ENABLED", True)):
            return False
            
        if self.redis is None:
            return False
            
        if self.mode == "REDISEARCH" and not _REDISEARCH_CLIENT_AVAILABLE:
            return False
            
        return True

    def _has_redisearch(self) -> bool:
        """Check if the Redis server has RediSearch module enabled.
        
        Returns:
            bool: True if RediSearch is available, False otherwise
        """
        if not _REDISEARCH_CLIENT_AVAILABLE:
            return False
            
        try:
            # Execute a simple RediSearch command to verify it's available
            self.redis.execute_command("FT._LIST")
            return True
        except Exception:
            return False

    def _ensure_dim(self) -> int:
        """Ensure the embedding dimension is known by making a test embedding if needed.
        
        Returns:
            int: The dimension of the embedding vectors
        """
        if self._dim is None:
            # Make a test embedding to determine the vector dimension
            v = _embeddings.embed_query("dimension_probe")
            self._dim = len(v)
        return self._dim

    def _ensure_lsh_hyperplanes(self) -> List[List[float]]:
        """Ensure LSH hyperplanes are initialized.
        
        Generates random hyperplanes for LSH bucketing if not already done.
        
        Returns:
            List of hyperplane vectors, each of dimension equal to the embedding size
        """
        if self._lsh_hyperplanes is not None:
            return self._lsh_hyperplanes

        dim = self._ensure_dim()
        rng = random.Random(self.lsh_seed)  # Seeded for reproducibility
        
        # Generate random hyperplanes with normally distributed components
        planes: List[List[float]] = []
        for _ in range(max(self.lsh_num_bits, 1)):  # At least 1 hyperplane
            # Each hyperplane is a random vector with components from N(0,1)
            planes.append([rng.gauss(0.0, 1.0) for _ in range(dim)])
            
        self._lsh_hyperplanes = planes
        return planes

    def _lsh_bucket_id(self, vec: List[float]) -> str:
        """Compute the LSH bucket ID for a vector.
        
        Uses random hyperplane hashing to map similar vectors to the same bucket
        with high probability.
        
        Args:
            vec: The vector to hash
            
        Returns:
            A hexadecimal string representing the bucket ID
        """
        planes = self._ensure_lsh_hyperplanes()
        bits = 0
        
        # For each hyperplane, determine which side of it the vector is on
        for i, p in enumerate(planes):
            if _dot(vec, p) >= 0.0:
                bits |= (1 << i)  # Set the i-th bit if on the positive side
                
        # Convert the bit vector to a hexadecimal string
        width = max(1, (len(planes) + 3) // 4)  # Number of hex digits needed
        return f"{bits:0{width}x}"

    def _bucket_zset_key(self, bucket_id: str) -> str:
        """Get the Redis key for an LSH bucket's LRU sorted set.
        
        Args:
            bucket_id: The LSH bucket ID
            
        Returns:
            Redis key for the bucket's LRU sorted set
        """
        return f"{config.PROXIMITY_CACHE_KEY_PREFIX}bucket:{bucket_id}:__lru__"

    def ensure_index(self) -> None:
        """Ensure the RediSearch index exists (REDISEARCH mode only).
        
        Creates the index if it doesn't exist. This is a no-op for non-REDISEARCH modes.
        """
        if self.mode != "REDISEARCH":
            return
            
        if not self.is_available():
            self._log_disabled_reason()
            return

        # Check if RediSearch is available on the server
        if not self._has_redisearch():
            if not self._disabled_reason_logged:
                self._disabled_reason_logged = True
                logging.info(
                    "Proximity cache disabled (Redis server has no RediSearch). "
                    "Run Redis Stack / redis-stack-server."
                )
            return

        # Check if index already exists
        try:
            self.redis.ft(self.index_name).info()
            return  # Index exists, nothing to do
        except Exception:
            pass  # Index doesn't exist, will create it

        # Get the embedding dimension
        dim = self._ensure_dim()
        
        # Define the RediSearch schema
        schema = (
            # Original query text (for debugging)
            TextField("query"),
            
            # JSON-serialized list of document texts
            TextField("docs_json"),
            
            # JSON-serialized list of base64-encoded document embeddings
            TextField("doc_embeddings_json"),
            
            # Timestamps for TTL and LRU
            NumericField("created_at"),
            NumericField("last_access"),
            
            # Vector field for approximate nearest neighbor search
            VectorField(
                "query_embedding",  # Field name
                "HNSW",            # Index type (Hierarchical Navigable Small World)
                {
                    "TYPE": "FLOAT32",
                    "DIM": dim,    # Dimension of the embedding vectors
                    "DISTANCE_METRIC": "COSINE",  # Distance metric for ANN search
                    "M": 16,                      # Max number of outgoing edges in HNSW graph
                    "EF_CONSTRUCTION": 200,       # Size of the dynamic candidate list
                },
            ),
        )

        # Create the index with a prefix to scope the keys
        definition = IndexDefinition(
            prefix=[config.PROXIMITY_CACHE_KEY_PREFIX], 
            index_type=IndexType.HASH
        )
        self.redis.ft(self.index_name).create_index(schema, definition=definition)
        
        # Log success (once per instance)
        if not self._index_ready_logged:
            self._index_ready_logged = True
            logging.info(f"Proximity cache index ensured: {self.index_name}")

    def _touch_lru(self, key: str) -> None:
        """Update the last access time for a key in the global LRU.
        
        Args:
            key: The cache key to update
        """
        if self.eviction_policy != "LRU":
            return
            
        now = time.time()
        
        # Update the global LRU sorted set
        self.redis.zadd(self._lru_zset_key, {key: now})
        
        # Also update the last_access field in the hash (if it exists)
        try:
            self.redis.hset(key, "last_access", now)
        except Exception:
            pass  # Ignore if the key doesn't exist

    def _touch_bucket(self, bucket_zset_key: str, key: str) -> None:
        """Update the last access time for a key in an LSH bucket's LRU.
        
        Args:
            bucket_zset_key: The Redis key for the bucket's LRU sorted set
            key: The cache key to update
        """
        if self.eviction_policy != "LRU":
            return
            
        now = time.time()
        
        # Update the bucket's LRU sorted set
        self.redis.zadd(bucket_zset_key, {key: now})
        
        # Also update the last_access field in the hash (if it exists)
        try:
            self.redis.hset(key, "last_access", now)
        except Exception:
            pass  # Ignore if the key doesn't exist

    def _evict_if_needed(self) -> None:
        """Evict entries if we're over capacity (FLAT and REDISEARCH modes).
        
        Removes the least recently used entries to maintain the cache at or below capacity.
        """
        if self.capacity <= 0:  # No capacity limit
            return
            
        if self.eviction_policy not in {"LRU", "FIFO"}:
            return  # Only LRU and FIFO eviction are supported

        try:
            # Check current cache size
            size = int(self.redis.zcard(self._lru_zset_key))
            if size <= self.capacity:
                return  # Under capacity, nothing to do
                
            # Calculate how many items to evict
            overflow = size - self.capacity
            
            # Get the oldest entries (by score, which is the timestamp)
            victims = self.redis.zrange(self._lru_zset_key, 0, overflow - 1)
            if not victims:
                return
                
            # Use a pipeline for atomic deletion
            pipe = self.redis.pipeline()
            for v in victims:
                # Convert to string if it's bytes (redis-py < 3.0 compatibility)
                k = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
                pipe.delete(k)  # Delete the hash
                pipe.zrem(self._lru_zset_key, k)  # Remove from LRU sorted set
                
            # Execute all deletions in a single atomic operation
            pipe.execute()
            
        except Exception as e:
            logging.error(f"Proximity cache eviction error: {e}")

    def _evict_bucket_if_needed(self, bucket_zset_key: str) -> None:
        """Evict entries from an LSH bucket if it's over capacity.
        
        Args:
            bucket_zset_key: The Redis key for the bucket's LRU sorted set
        """
        if self.lsh_bucket_capacity <= 0:
            return  # No capacity limit for buckets

        try:
            # Check current bucket size
            size = int(self.redis.zcard(bucket_zset_key))
            if size <= self.lsh_bucket_capacity:
                return  # Under capacity, nothing to do
                
            # Calculate how many items to evict
            overflow = size - self.lsh_bucket_capacity
            
            # Get the oldest entries (by score, which is the timestamp)
            victims = self.redis.zrange(bucket_zset_key, 0, overflow - 1)
            if not victims:
                return
                
            # Use a pipeline for atomic deletion
            pipe = self.redis.pipeline()
            for v in victims:
                # Convert to string if it's bytes (redis-py < 3.0 compatibility)
                k = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
                pipe.delete(k)  # Delete the hash
                pipe.zrem(bucket_zset_key, k)  # Remove from bucket's LRU
                
            # Execute all deletions in a single atomic operation
            pipe.execute()
            
        except Exception as e:
            logging.error(f"Proximity cache bucket eviction error: {e}")

    def _read_docs_from_entry(
        self,
        query_vec: list[float],
        entry_docs_json: str,
        entry_doc_embeddings_json: Optional[str],
    ) -> list[str] | None:
        """Extract and optionally re-rank documents from a cache entry.
        
        This helper function parses the cached documents and their embeddings, and
        optionally re-ranks them based on their similarity to the query vector.
        
        Args:
            query_vec: The query embedding vector
            entry_docs_json: JSON string containing the list of document texts
            entry_doc_embeddings_json: Optional JSON string containing base64-encoded document embeddings
            
        Returns:
            List of document texts, potentially re-ordered by relevance to the query
        """
        try:
            docs = json.loads(entry_docs_json)
            if not isinstance(docs, list) or not docs:
                return None

            if entry_doc_embeddings_json:
                encoded = json.loads(entry_doc_embeddings_json)
                if isinstance(encoded, list) and encoded:
                    decoded = [base64.b64decode(s.encode("utf-8")) for s in encoded]
                    doc_vecs = [_from_float32_bytes(b) for b in decoded]
                    scored = [(_cosine_distance(query_vec, dv), d) for dv, d in zip(doc_vecs, docs)]
                    scored.sort(key=lambda x: x[0])
                    docs = [d for _, d in scored]

            return docs[: int(self.top_k)]
        except Exception:
            return None

    def get_context(
        self, query: str, top_k: Optional[int] = None, tolerance: Optional[float] = None, rerank_factor: Optional[int] = None
    ) -> Tuple[Optional[List[str]], Optional[float]]:
        """Retrieve cached documents that are semantically similar to the query.
        
        This is the main entry point for the cache lookup. It first checks if the cache
        is available, then delegates to the appropriate method based on the cache mode.
        
        Args:
            query: The input query text
            top_k: Number of documents to return (overrides default if provided)
            tolerance: Maximum cosine distance for a cache hit (overrides default if provided)
            rerank_factor: Factor for over-fetching documents (overrides default if provided)
            
        Returns:
            A tuple of (list of document texts, distance to query) if found, or (None, None) on miss
        """
        if not self.is_available():
            self._log_disabled_reason()
            return None, None

        # Use provided parameters or fall back to instance defaults
        top_k = top_k or self.top_k
        tolerance = tolerance or self.tolerance
        rerank_factor = rerank_factor or self.rerank_factor

        try:
            # Generate embedding for the query
            query_embedding = _embeddings.embed_query(query)
            self._ensure_dim()  # Ensure embedding dimension is known

            # Delegate to the appropriate method based on cache mode
            if self.mode == "REDISEARCH":
                return self._get_context_redis(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    tolerance=tolerance,
                    rerank_factor=rerank_factor,
                )
            elif self.mode == "LSH":
                return self._get_context_lsh(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    tolerance=tolerance,
                    rerank_factor=rerank_factor,
                )
            else:  # FLAT mode (default fallback)
                return self._get_context_flat(
                    query=query,
                    query_embedding=query_embedding,
                    top_k=top_k,
                    tolerance=tolerance,
                    rerank_factor=rerank_factor,
                )
        except Exception as e:
            logging.error(f"Proximity cache get_context error: {e}")
            return None, None

    def _get_context_flat(
        self, query: str, query_embedding: List[float], top_k: int, tolerance: float, rerank_factor: int
    ) -> Tuple[Optional[List[str]], Optional[float]]:
        """Get context using a linear scan over all entries (Proximity-FLAT).
        
        This implements the Proximity-FLAT algorithm from the paper, which performs
        a linear scan over all cached entries to find the closest match.
        
        Args:
            query: The input query text (for debugging/logging)
            query_embedding: The embedding vector of the query
            top_k: Number of documents to return
            tolerance: Maximum cosine distance for a cache hit
            rerank_factor: Factor for over-fetching documents before re-ranking
            
        Returns:
            A tuple of (list of document texts, distance to query) if found, or (None, None) on miss
        """
        if not self.redis:
            return None, None

        # Get all keys from the global LRU zset (this tracks all entries in the cache)
        keys = self.redis.zrange(self._lru_zset_key, 0, -1)
        if not keys:
            return None, None  # Cache is empty

        # Over-fetch more candidates for re-ranking (ρ from the paper)
        overfetch_k = min(len(keys), max(top_k, top_k * max(rerank_factor, 1)))
        candidates = []

        # Get all entries in a pipeline for efficiency (reduces round trips to Redis)
        pipe = self.redis.pipeline()
        for k in keys:
            pipe.hgetall(k)  # Get the full hash for each key
        entries = pipe.execute()

        # Calculate distances to find the most similar cached query
        for k, entry in zip(keys, entries):
            if not entry:
                continue  # Skip missing entries
                
            try:
                # Decode the cached query embedding
                cached_embedding = _from_float32_bytes(base64.b64decode(entry[b"query_embedding"]))
                
                # Calculate cosine distance to the query
                distance = _cosine_distance(query_embedding, cached_embedding)
                
                # If within tolerance, add to candidates
                if distance <= tolerance:
                    candidates.append((distance, entry, k))
                    
            except Exception as e:
                logging.warning(f"Error processing cache entry: {e}")
                continue

        # Sort candidates by distance (closest first) and take top overfetch_k
        candidates.sort(key=lambda x: x[0])
        best_candidates = candidates[:overfetch_k]

        if not best_candidates:
            return None, None  # No matches within tolerance

        # Get the best match (smallest distance)
        best_distance, best_entry, best_key = best_candidates[0]
        
        # Update LRU for the best match (mark as recently used)
        self._touch_lru(best_key.decode("utf-8") if isinstance(best_key, (bytes, bytearray)) else str(best_key))

        # Read and return the documents (truncating to top_k)
        docs, _ = self._read_docs_from_entry(best_entry)
        return docs[:top_k], best_distance

    def _get_context_redis(
        self, query: str, query_embedding: List[float], top_k: int, tolerance: float, rerank_factor: int
    ) -> Tuple[Optional[List[str]], Optional[float]]:
        """Get context using RediSearch (Proximity-REDISEARCH).
        
        This implements the Proximity-REDISEARCH algorithm from the paper, which uses
        RediSearch to efficiently search for the closest match.
        
        Args:
            query: The input query text (for debugging/logging)
            query_embedding: The embedding vector of the query
            top_k: Number of documents to return
            tolerance: Maximum cosine distance for a cache hit
            rerank_factor: Factor for over-fetching documents before re-ranking
            
        Returns:
            A tuple of (list of document texts, distance to query) if found, or (None, None) on miss
        """
        if not self.redis:
            return None, None

        # Create a RediSearch query to find the closest match
        query_vec_bytes = _to_float32_bytes(query_embedding)
        q = (
            Query(f"*=>[KNN 1 @query_embedding $vec AS score]")
            .sort_by("score")
            .return_fields("docs_json", "doc_embeddings_json", "score")
            .dialect(2)
        )
        res = self.redis.ft(self.index_name).search(q, query_params={"vec": query_vec_bytes})
        if not res.docs:
            return None, None

        # Get the best match (smallest distance)
        best = res.docs[0]
        best_distance = float(best.score)
        if best_distance > tolerance:
            return None, best_distance

        # Update LRU for the best match (mark as recently used)
        best_id = getattr(best, "id", None)
        if best_id and self.eviction_policy == "LRU":
            self._touch_lru(best_id)

        # Read and return the documents (truncating to top_k)
        docs_json = getattr(best, "docs_json", None)
        if not docs_json:
            return None, best_distance

        docs = self._read_docs_from_entry(query_embedding, docs_json, getattr(best, "doc_embeddings_json", None))
        return docs[:top_k], best_distance

    def _get_context_lsh(
        self, query: str, query_embedding: List[float], top_k: int, tolerance: float, rerank_factor: int
    ) -> Tuple[Optional[List[str]], Optional[float]]:
        """Get context using Locality-Sensitive Hashing (Proximity-LSH).
        
        This implements the Proximity-LSH algorithm from the paper, which uses
        Locality-Sensitive Hashing to efficiently search for the closest match.
        
        Args:
            query: The input query text (for debugging/logging)
            query_embedding: The embedding vector of the query
            top_k: Number of documents to return
            tolerance: Maximum cosine distance for a cache hit
            rerank_factor: Factor for over-fetching documents before re-ranking
            
        Returns:
            A tuple of (list of document texts, distance to query) if found, or (None, None) on miss
        """
        if not self.redis:
            return None, None

        # Get the bucket ID for the query
        bucket_id = self._lsh_bucket_id(query_embedding)
        bucket_zset_key = self._bucket_zset_key(bucket_id)

        # Get all keys from the bucket's LRU zset
        keys = self.redis.zrange(bucket_zset_key, 0, -1)
        if not keys:
            return None, None  # Bucket is empty

        # Over-fetch more candidates for re-ranking (ρ from the paper)
        overfetch_k = min(len(keys), max(top_k, top_k * max(rerank_factor, 1)))
        candidates = []

        # Get all entries in a pipeline for efficiency (reduces round trips to Redis)
        pipe = self.redis.pipeline()
        for k in keys:
            pipe.hgetall(k)  # Get the full hash for each key
        entries = pipe.execute()

        # Calculate distances to find the most similar cached query
        for k, entry in zip(keys, entries):
            if not entry:
                continue  # Skip missing entries
                
            try:
                # Decode the cached query embedding
                cached_embedding = _from_float32_bytes(base64.b64decode(entry[b"query_embedding"]))
                
                # Calculate cosine distance to the query
                distance = _cosine_distance(query_embedding, cached_embedding)
                
                # If within tolerance, add to candidates
                if distance <= tolerance:
                    candidates.append((distance, entry, k))
                    
            except Exception as e:
                logging.warning(f"Error processing cache entry: {e}")
                continue

        # Sort candidates by distance (closest first) and take top overfetch_k
        candidates.sort(key=lambda x: x[0])
        best_candidates = candidates[:overfetch_k]

        if not best_candidates:
            return None, None  # No matches within tolerance

        # Get the best match (smallest distance)
        best_distance, best_entry, best_key = best_candidates[0]
        
        # Update LRU for the best match (mark as recently used)
        self._touch_bucket(bucket_zset_key, best_key)

        # Read and return the documents (truncating to top_k)
        docs_json = best_entry.get(b"docs_json", None)
        if not docs_json:
            return None, best_distance

        docs = self._read_docs_from_entry(query_embedding, docs_json.decode("utf-8"), best_entry.get(b"doc_embeddings_json", None))
        return docs[:top_k], best_distance

    def set_context(
        self, query: str, docs: List[str], doc_embeddings: List[List[float]]
    ) -> bool:
        """Store documents and their embeddings in the cache.
        
        This method adds a new entry to the cache with the query, its embedding,
        and the associated documents with their embeddings.
        
        Args:
            query: The original query text
            docs: List of document texts to cache
            doc_embeddings: List of embedding vectors corresponding to the documents
            
        Returns:
            bool: True if the operation was successful, False otherwise
        """
        if not self.is_available() or not docs or not doc_embeddings:
            return False

        try:
            # Generate embedding for the query
            query_embedding = _embeddings.embed_query(query)
            self._ensure_dim()  # Ensure embedding dimension is known

            # Prepare the data for storage
            key = _stable_key(query)  # Generate a stable cache key
            now = time.time()
            
            # Encode the query embedding as base64 for storage
            query_embedding_bytes = _to_float32_bytes(query_embedding)
            query_embedding_b64 = base64.b64encode(query_embedding_bytes).decode("utf-8")
            
            # Encode document embeddings as base64 for storage
            doc_embeddings_b64 = [
                base64.b64encode(_to_float32_bytes(emb)).decode("utf-8") 
                for emb in doc_embeddings
            ]
            
            # Create a Redis hash with all the data
            pipe = self.redis.pipeline()
            pipe.hset(
                key,
                mapping={
                    "query": query,  # Original query text (for debugging)
                    "query_embedding": query_embedding_b64,  # Base64-encoded embedding
                    "docs_json": json.dumps(docs),  # JSON-serialized documents
                    "doc_embeddings_json": json.dumps(doc_embeddings_b64),  # JSON-serialized embeddings
                    "created_at": now,  # Creation timestamp
                    "last_access": now,  # Last access timestamp (for LRU)
                },
            )
            # Set TTL on the hash
            pipe.expire(key, self.ttl_seconds)

            # Update the appropriate data structures based on the cache mode
            if self.mode == "LSH":
                # For LSH mode, add to the appropriate bucket
                bucket_id = self._lsh_bucket_id(query_embedding)
                bucket_zset_key = self._bucket_zset_key(bucket_id)
                pipe.zadd(bucket_zset_key, {key: now})  # Add to bucket's LRU
                pipe.expire(bucket_zset_key, self.ttl_seconds)  # Set TTL on the bucket
                pipe.execute()  # Execute all commands in the pipeline
                
                # Evict old entries if the bucket is over capacity
                self._evict_bucket_if_needed(bucket_zset_key)
                
            else:  # FLAT or REDISEARCH mode
                # For these modes, just use the global LRU
                pipe.zadd(self._lru_zset_key, {key: now})  # Add to global LRU
                pipe.execute()  # Execute all commands in the pipeline
                
                # Evict old entries if we're over capacity
                self._evict_if_needed()

            # If using REDISEARCH, ensure the document is indexed
            if self.mode == "REDISEARCH":
                self.ensure_index()  # Creates the index if it doesn't exist

            return True

        except Exception as e:
            logging.error(f"Proximity cache set_context error: {e}")
            return False

    def _read_docs_from_entry(self, entry: dict) -> Tuple[List[str], List[List[float]]]:
        """Extract documents and their embeddings from a cache entry.
        
        Args:
            entry: A Redis hash containing the cached data
            
        Returns:
            A tuple of (list of document texts, list of document embeddings)
        """
        # Parse the JSON-serialized documents
        docs = json.loads(entry.get(b"docs_json", b"[]").decode("utf-8"))
        
        # Parse and decode the base64-encoded document embeddings
        doc_embeddings = [
            _from_float32_bytes(base64.b64decode(emb)) 
            for emb in json.loads(entry.get(b"doc_embeddings_json", b"[]").decode("utf-8"))
        ]
        
        return docs, doc_embeddings

    def _lsh_bucket_id(self, query_embedding: List[float]) -> int:
        """Get the LSH bucket ID for a query embedding.
        
        This method uses the first 32 bits of the query embedding's hash as the bucket ID.
        
        Args:
            query_embedding: The query embedding vector
            
        Returns:
            int: The LSH bucket ID
        """
        # Calculate the hash of the query embedding
        query_hash = int(hashlib.md5(_to_float32_bytes(query_embedding)).hexdigest(), 16)
        
        # Use the first 32 bits as the bucket ID
        bucket_id = query_hash & 0xFFFFFFFF
        
        return bucket_id

    def _bucket_zset_key(self, bucket_id: int) -> str:
        """Get the Redis key for an LSH bucket's LRU sorted set.
        
        Args:
            bucket_id: The LSH bucket ID
            
        Returns:
            str: The Redis key for the bucket's LRU
        """
        return f"{config.PROXIMITY_CACHE_KEY_PREFIX}:bucket:{bucket_id}:__lru__"

    def _ensure_dim(self) -> int:
        """Ensure the embedding dimension is known.
        
        This method checks if the embedding dimension is already known, and if not,
        it tries to infer it from the cache entries.
        
        Returns:
            int: The embedding dimension
        """
        if self.dim is not None:
            return self.dim
        
        # Try to infer the dimension from the cache entries
        keys = self.redis.zrange(self._lru_zset_key, 0, -1)
        for k in keys:
            entry = self.redis.hgetall(k)
            if entry:
                query_embedding = entry.get(b"query_embedding")
                if query_embedding:
                    self.dim = len(_from_float32_bytes(base64.b64decode(query_embedding)))
                    return self.dim
        
        # If still unknown, raise an error
        raise ValueError("Embedding dimension is unknown")

    def _log_disabled_reason(self) -> None:
        """Log the reason why the cache is disabled.
        
        This method logs a message explaining why the cache is disabled, based on the
        current configuration and environment.
        """
        if not self.redis:
            logging.info("Proximity cache disabled (no Redis connection).")
        elif self.mode == "REDISEARCH" and not self._has_redisearch():
            logging.info("Proximity cache disabled (Redis server has no RediSearch).")
        else:
            logging.info("Proximity cache disabled (unknown reason).")

    def _has_redisearch(self) -> bool:
        """Check if RediSearch is available on the Redis server.
        
        Returns:
            bool: True if RediSearch is available, False otherwise
        """
        try:
            self.redis.ft(self.index_name).info()
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        """Check if the cache is available.
        
        This method checks if the cache is enabled, and if the Redis connection is available.
        
        Returns:
            bool: True if the cache is available, False otherwise
        """
        return self.enabled and self.redis

    def _stable_key(self, query: str) -> str:
        """Generate a stable cache key for a query.
        
        This method generates a stable cache key by hashing the query text and taking
        the first 32 characters of the hash.
        
        Args:
            query: The query text
            
        Returns:
            str: The stable cache key
        """
        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
        return f"{config.PROXIMITY_CACHE_KEY_PREFIX}:{query_hash[:32]}"

def proximity_cache_get_context(query: str) -> Tuple[Optional[List[str]], Optional[float]]:
    """Get cached context for a query if a similar query exists in the cache.
    
    This is the main entry point for retrieving documents from the proximity cache.
    It uses the singleton instance of ProximityRetrievalCache to find and return
    documents that were previously stored for a similar query.
    
    Args:
        query: The input query text
        
    Returns:
        A tuple of (list of document texts, distance to query) if found, 
        or (None, None) if no match is found or cache is disabled
    """
    return _proximity_cache_singleton.get_context(query)

def proximity_cache_set_context(query: str, docs: List[str]) -> bool:
    """Store documents in the cache for future retrieval by similar queries.
    
    This is the main entry point for storing documents in the proximity cache.
    It uses the singleton instance of ProximityRetrievalCache to store the
    documents along with their embeddings for future similarity-based retrieval.
    
    Args:
        query: The query text to associate with the documents
        docs: List of document texts to store in the cache
        
    Returns:
        bool: True if the operation was successful, False otherwise
    """
    # Generate embeddings for the documents
    doc_embeddings = _embeddings.embed_documents(docs)
    
    # Store in the cache
    return _proximity_cache_singleton.set_context(query, docs, doc_embeddings)

# Initialize the singleton instance after the class is defined
_proximity_cache_singleton = ProximityRetrievalCache(
    redis_url=config.REDIS_URL,
    index_name=config.PROXIMITY_CACHE_INDEX_NAME,
    ttl_seconds=config.CACHE_TTL_SECONDS,
    capacity=config.PROXIMITY_CACHE_CAPACITY,
    tolerance=config.PROXIMITY_DISTANCE_TOLERANCE,
    top_k=config.PROXIMITY_RETRIEVAL_TOP_K,
    rerank_factor=config.PROXIMITY_RERANK_FACTOR,
    eviction_policy=config.PROXIMITY_EVICTION_POLICY,
)
