from fastapi import FastAPI
from pydantic import BaseModel
import json
import uuid
from datetime import datetime
from typing import Optional, List
import redis
import os

app = FastAPI(title="Empathetic Chatbot API")
try:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    redis_client=redis.Redis(host=REDIS_HOST, port=6379, db=0)
    # redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()
    print("✅ Connected to Redis successfully!")
except redis.exceptions.ConnectionError:
    redis_client = None
    print("❌ Could not connect to Redis. Make sure Redis server is running.")
    
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Empathetic Chatbot API",
        "status": "running",
        "redis_connected": redis_client is not None
    }
    
class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[str]] = None
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = None
    
class ChatResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ResultResponse(BaseModel):
    job_id: str
    status: str
    response: Optional[str] = None
    emotion: Optional[str] = None
    safety_triggered: Optional[bool] = None
    severity: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


@app.post("/chat", response_model=ChatResponse)
async def push_request(request: ChatRequest):
    """Endpoint to submit a chat message for processing"""
    job_id = str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    
    job_data = {
        "job_id": job_id,
        "user_id": request.user_id,
        "session_id": session_id,
        "message": request.message,
        "conversation_history": request.conversation_history or [],
        "status": "pending",
        "created_at": datetime.now().isoformat()
    }
    # Push job to Redis queue
    redis_client.rpush("chat_jobs", json.dumps(job_data))
    
    return ChatResponse(
        job_id=job_id,
        status="submitted",
        message="Your message has been submitted for processing."
    )

@app.get("/response/{job_id}", response_model=ResultResponse)
async def get_response(job_id: str):
    if job_id is None:
        return ResultResponse(
            job_id=job_id,
            status="error",
            response=None
        )
    """Endpoint to retrieve the response for a given job_id"""
    result_key = f"chat_result:{job_id}"
    result_data = redis_client.get(result_key)
    if result_data is None:
        return ResultResponse(
            job_id=job_id,
            status="pending",
            response=None
        )
    result = json.loads(result_data)
    return ResultResponse(**result)