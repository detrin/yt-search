import os
import uuid
import subprocess
import base64
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from redis import Redis
from pydantic import BaseModel

app = FastAPI()
security = HTTPBasic()
redis = Redis(host="redis", port=6379, decode_responses=True)

class SearchRequest(BaseModel):
    question: str

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("API_USER", "admin")
    correct_password = os.getenv("API_PASS", "secret")
    
    if not (credentials.username == correct_username and 
            credentials.password == correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.post("/v1/search")
async def create_search(request: SearchRequest, user: str = Depends(get_current_user)):
    job_id = str(uuid.uuid4())
    redis.set(job_id, "processing")
    
    # Run processing in background
    subprocess.Popen([
        "bash", "run.sh",
        request.question
    ], env={
        **os.environ,
        "JOB_ID": job_id,
        "REDIS_HOST": "redis"
    })
    
    return {"job_id": job_id}

@app.get("/v1/status/{job_id}")
async def get_status(job_id: str):
    status = redis.get(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": status}

@app.get("/v1/results/{job_id}")
async def get_results(job_id: str):
    # Get raw bytes from Redis
    result_bytes = redis.get(f"{job_id}:result")
    status = redis.get(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    if status != "done":  # Decode status if needed
        return {"status": "processing"}
    
    if not result_bytes:
        raise HTTPException(status_code=500, detail="Result missing")
    
    try:
        # Base64 decode the result and convert to string
        decoded_result = base64.b64decode(result_bytes).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding error: {str(e)}")
    
    return {"result": decoded_result}