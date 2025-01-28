import redis
import os

def update_redis_status(job_id: str, status: str, result: str = None):
    r = redis.Redis(host="redis", port=6379, decode_responses=True)
    r.set(job_id, status)
    if result:
        r.set(f"{job_id}:result", result)