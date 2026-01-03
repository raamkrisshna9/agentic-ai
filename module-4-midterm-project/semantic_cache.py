#####################
# semantic_cache.py
#####################
"""Semantic cache (Redis + RediSearch vector KNN).

This module implements a production-style semantic cache:

- Inputs: a user query string
- Key idea: cache by semantic similarity (vector embedding), not exact string match
- Storage: Redis HASH documents (one per cached query)
- Retrieval: RediSearch KNN query against an HNSW vector index

Redis schema per cached item (HASH):

- query: normalized query text (lowercased, whitespace-collapsed)
- response: model response string
- created_at: unix timestamp
- embedding: FLOAT32 bytes (little-endian) of the query embedding

Dependencies / runtime requirements:

- redis-py with RediSearch client modules (redis>=5.0.0)
- Redis Stack / RediSearch enabled on the server (FT.* commands available)

If either the client modules or server RediSearch are unavailable, semantic caching is
automatically disabled and the rest of the application continues to work.
"""

import hashlib
import time
import logging
import struct
import redis

# RediSearch client imports.
#
# Different redis-py versions expose slightly different module paths, so we try both.
# If these imports fail, semantic caching is disabled (but the rest of the app keeps running).
try:
    from redis.commands.search.field import TextField, NumericField, VectorField
    from redis.commands.search.query import Query
    try:
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    except Exception:
        from redis.commands.search.index_definition import IndexDefinition, IndexType
    _REDISEARCH_CLIENT_AVAILABLE = True
except Exception:
    TextField = NumericField = VectorField = None
    IndexDefinition = IndexType = None
    Query = None
    _REDISEARCH_CLIENT_AVAILABLE = False

import config
from langchain_openai import OpenAIEmbeddings

_embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

# Normalize the query.
#
# Purpose:
# - stabilize cache keys across minor whitespace/case differences
# - reduce accidental cache misses (e.g., "Agentic  AI" vs "agentic ai")
def _normalize_query(q: str) -> str:
    return " ".join((q or "").strip().lower().split())

# Generate a stable key for the query.
#
# We use a SHA-256 hash of the normalized query to:
# - avoid extremely long keys
# - avoid storing raw queries in keys
# - have deterministic keys for exact lookup / overwrites
def _stable_key(q: str) -> str:
    nq = _normalize_query(q)
    digest = hashlib.sha256(nq.encode("utf-8")).hexdigest()
    return f"{config.SEMANTIC_CACHE_KEY_PREFIX}{digest}"


# Convert a float vector to raw bytes.
#
# Purpose:
# - RediSearch vector fields require raw bytes.
# - Return bytes in little-endian float32 format to match Redis vector index expectations. e.g: <3f
def _to_float32_bytes(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)

# SemanticCache class
class SemanticCache:
    """Redis-backed semantic cache.

    This class is responsible for:

    - connecting to Redis
    - creating the RediSearch index (one-time)
    - storing (query -> embedding + response)
    - retrieving via KNN (embedding similarity)

    Notes:
    - The cache uses COSINE distance. Lower distance => more similar.
    - A cache hit is accepted when best_distance <= max_distance.
    """

    # Initialize the semantic cache
    def __init__(
        self,
        redis_url: str = config.REDIS_URL,
        index_name: str = config.SEMANTIC_CACHE_INDEX_NAME,
        ttl_seconds: int = config.CACHE_TTL_SECONDS,
        top_k: int = config.SEMANTIC_CACHE_TOP_K,
        max_distance: float = config.SEMANTIC_CACHE_MAX_DISTANCE,
    ):
        self.redis_url = redis_url
        self.index_name = index_name
        self.ttl_seconds = ttl_seconds
        self.top_k = top_k
        self.max_distance = max_distance

        # Redis connection.
        # We keep a local client instance and disable semantic caching if the connection fails.
        try:
            self.redis = redis.Redis.from_url(redis_url)
            self.redis.ping()
        except Exception as e:
            logging.error(f"Semantic cache Redis connection error: {e}")
            self.redis = None

        # Cached embedding dimension.
        # This is resolved lazily from the embedding model (first use) so it stays correct
        # even if you change EMBEDDING_MODEL.
        self._dim = None

        # Logging flags to avoid noisy repeated logs.
        self._disabled_reason_logged = False
        self._index_ready_logged = False


    # Check if the semantic cache is available
    #
    # Semantic cache is available only if:
    # - enabled via config
    # - Redis connection is alive
    # - redis-py has RediSearch client modules
    def is_semantic_cache_available(self) -> bool:
        return (
            bool(getattr(config, "SEMANTIC_CACHE_ENABLED", True))
            and self.redis is not None
            and _REDISEARCH_CLIENT_AVAILABLE
        )

    # Log a single, actionable message about why semantic caching is disabled.
    # This helps debugging cases where the semantic cache always returns MISS with distance=None.
    def _log_disabled_reason(self) -> None:
        if self._disabled_reason_logged:
            return
        self._disabled_reason_logged = True

        if not bool(getattr(config, "SEMANTIC_CACHE_ENABLED", True)):
            logging.info("Semantic cache disabled by config (SEMANTIC_CACHE_ENABLED=False)")
            return
        if self.redis is None:
            logging.info("Semantic cache disabled (Redis connection not available)")
            return
        if not _REDISEARCH_CLIENT_AVAILABLE:
            logging.info(
                "Semantic cache disabled (redis-py RediSearch client not available). "
                "Upgrade dependency: pip install -U 'redis>=5.0.0'"
            )
            return


    # Server-side capability check
    # FT._LIST is a RediSearch command; if it fails, Redis likely isn't Redis Stack.
    def _has_redisearch(self) -> bool:
        if not _REDISEARCH_CLIENT_AVAILABLE:
            return False
        try:
            self.redis.execute_command("FT._LIST")
            logging.info("Redisearch is available")
            return True
        except Exception:
            logging.info("Redisearch is not available")
            return False

    # Determine embedding vector dimension once.
    # RediSearch vector fields require DIM at index-creation time.
    def _ensure_dim(self) -> int:
        if self._dim is None:
            v = _embeddings.embed_query("dimension_probe")
            self._dim = len(v)
        return self._dim

    # Create the semantic cache index if it doesn't exist.
    # This is safe to call repeatedly.
    def ensure_redisearch_index(self) -> None:
        if not self.is_semantic_cache_available():
            self._log_disabled_reason()
            return

        if not self._has_redisearch():
            if not self._disabled_reason_logged:
                self._disabled_reason_logged = True
                logging.info(
                    "Semantic cache disabled (Redis server has no RediSearch). "
                    "Run Redis Stack / redis-stack-server."
                )
            return

        try:
            self.redis.ft(self.index_name).info()
            return
        except Exception:
            pass

        # Index schema:
        # - query/response: stored fields for inspection and retrieval
        # - created_at: optional metadata field
        # - embedding: vector field used by KNN, which is used to find the most similar documents
        # - HSNW, format for the vector field:
        #   - TYPE: FLOAT32
        #   - DIM: dimension of the embedding vector
        #   - DISTANCE_METRIC: cosine similarity
        #   - M: number of neighbors to consider
        #   - EF_CONSTRUCTION: number of neighbors to consider during index construction
        dim = self._ensure_dim()
        schema = (
            TextField("query"),
            TextField("response"),
            NumericField("created_at"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": dim,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
            ),
        )

        # Index definition
        definition = IndexDefinition(prefix=[config.SEMANTIC_CACHE_KEY_PREFIX], index_type=IndexType.HASH)
        
        # Create the index if it doesn't exist
        self.redis.ft(self.index_name).create_index(schema, definition=definition)
        if not self._index_ready_logged:
            self._index_ready_logged = True
            logging.info(f"Semantic cache index ensured: {self.index_name}")

    # Semantic lookup.
    # Returns (response, distance) if found within threshold.
    # Returns (None, distance) if nearest neighbor exists but is above threshold.
    # Returns (None, None) if semantic caching is disabled or no docs exist.
    def get(self, query: str) -> tuple[str | None, float | None]:
        if not self.is_semantic_cache_available():
            self._log_disabled_reason()
            return None, None

        try:
            # Ensure the index exists
            self.ensure_redisearch_index()
            if not self._has_redisearch():
                return None, None

            # Convert the incoming query to an embedding and then to bytes.
            # The RediSearch KNN query compares this vector against cached embeddings.
            embedding = _embeddings.embed_query(query)
            vec = _to_float32_bytes(embedding)

            # KNN query:
            # - KNN top_k: ask RediSearch for the closest top_k cached items
            # - score: RediSearch returns distance when using COSINE
            # - lower score => more similar
            # - Query for top_k nearest neighbors using the embedding vector with dialect 2 and sort by score and return response and score
            q = (
                Query(f"*=>[KNN {self.top_k} @embedding $vec AS score]")
                .sort_by("score")
                .return_fields("response", "score")
                .dialect(2)
            )

            # Search the index for the nearest neighbors
            res = self.redis.ft(self.index_name).search(q, query_params={"vec": vec})
            if not res.docs:
                return None, None
            # Get the best match document and distance
            best = res.docs[0]
            distance = float(best.score)

            # Thresholding.
            # max_distance is a safety knob: it prevents reusing a response for a loosely related query.
            if distance > self.max_distance:
                return None, distance
            # Return the best match document and distance
            return best.response, distance
        except Exception as e:
            logging.error(f"Semantic cache get error: {e}")
            return None, None

    # Store a semantic cache entry.
    # We always store using the normalized-query hash as the Redis key.
    # TTL is set per entry so old responses naturally expire.
    def set(self, query: str, response: str) -> None:
        if not self.is_semantic_cache_available():
            self._log_disabled_reason()
            return

        try:
            # Ensure the index exists, if not create it
            self.ensure_redisearch_index()
            if not self._has_redisearch():
                return

            # Embed the query and store both the raw bytes and human-readable fields.
            embedding = _embeddings.embed_query(query)
            key = _stable_key(query)
            now = time.time()

            # Store the semantic cache entry
            # Key is the normalized query hash and value is the response with metadata
            # The response metadata mapping is as follows:
            # - query: normalized query
            # - response: response
            # - created_at: timestamp
            # - embedding: embedding vector
            self.redis.hset(
                key,
                mapping={
                    "query": _normalize_query(query),
                    "response": response,
                    "created_at": now,
                    "embedding": _to_float32_bytes(embedding),
                },
            )
            self.redis.expire(key, int(self.ttl_seconds))
            logging.info(f"Semantic cache set: {key}")
        except Exception as e:
            logging.error(f"Semantic cache set error: {e}")


# Create a singleton instance of the SemanticCache
_semantic_cache_singleton = SemanticCache()

# Convenience wrapper used by main.py.
def semantic_cache_get(query: str) -> tuple[str | None, float | None]:
    return _semantic_cache_singleton.get(query)

# Convenience wrapper used by main.py.
def semantic_cache_set(query: str, response: str) -> None:
    _semantic_cache_singleton.set(query, response)
