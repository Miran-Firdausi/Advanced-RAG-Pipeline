import redis
import json

r = redis.Redis(host="localhost", port=6379, db=0)


def cache_summary(doc_hash: str, summary: str):
    r.set(f"doc:{doc_hash}:summary", summary)


def get_cached_data(doc_hash: str):
    return r.get(f"doc:{doc_hash}:summary")
