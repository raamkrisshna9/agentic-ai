##########################################################################################################
# cache_store.py
# This is a simple (key to key comparison) caching system using Redis.
# But in enterprises semantic cache is more powerful and used to cache the responses based on the context.
##########################################################################################################
""" This module provides functionality for simple caching responses (key to key comparison) using Redis. """
import config
import redis
import logging

REDIS_URL = config.REDIS_URL
CACHE_TTL_SECONDS = config.CACHE_TTL_SECONDS

# Initialize the Redis client
try:
    redis_client = redis.Redis.from_url(REDIS_URL)
    redis_client.ping()
    logging.info("Redis connection successful")
except Exception as e:
    logging.error(f"Redis connection Error: {e}")
    redis_client = None

# Redis cache key, which is used to store the response in the cache
def _cache_key(k: str) -> str:
    #rag:cache_key: is the prefix for the cache key
    return f"rag:cache_key: {k}"

# Get the cache from Redis, takes the cache key as input and returns the value
def get_cache(k: str):
    try:
        #.get() is used to get the value from the cache
        _value = redis_client.get(_cache_key(k))
        return _value.decode() if _value else None #.decode() is used to convert the bytes to string
    except Exception as e:
        logging.error(f"Redis get Error: {e}")
        return None

# Set the cache in Redis, takes the cache key and value as input and set the value in the cache with a ttl
def set_cache(k:str, v:str, ttl:int = CACHE_TTL_SECONDS):
    try:
        #.setex() is used to set the value in the cache with a ttl, common syntax is .setex(key, ttl, value)
        redis_client.setex(_cache_key(k), ttl, v)
    except Exception as e:
        logging.error(f"Redis set Error: {e}")

