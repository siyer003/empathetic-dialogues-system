import json
from datetime import datetime
import redis
from utils import generate_response, load_models
import os



try:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    redis_client=redis.Redis(host=REDIS_HOST, port=6379, db=0)
    # redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()
    print("✅ Connected to Redis successfully!")
except redis.exceptions.ConnectionError:
    redis_client = None
    print("❌ Could not connect to Redis. Make sure Redis server is running.")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")
load_models(model_path=MODEL_PATH)  # Load models at startup
# load_models(model_path="../models")

def initiate_chats(job_data):
    job_id = job_data["job_id"]
    message = job_data["message"]
    session_id = job_data["session_id"]
    conversation_history = job_data.get("conversation_history", [])
    job_data["status"] = "processing"
    
    redis_client.setex(f"job:{job_id}", 3600, json.dumps(job_data))
    
    conversation_history = []
    if session_id:
        history_key = f"session:{session_id}:history"
        history_data = redis_client.get(history_key)
        if history_data:
            conversation_history = json.loads(history_data)
            
    result = generate_response(message, conversation_history)
    
    if session_id:
        conversation_history.append(f"[USER] {message}")
        conversation_history.append(f"[BOT] {result['response']}")
        conversation_history = conversation_history[-10:]
        
        history_key = f"session:{session_id}:history"
        redis_client.setex(history_key, 3600, json.dumps(conversation_history))
    
    respone_data = {
        "job_id": job_id,
        "status": "completed",
        "response": result["response"],
        "emotion": result["emotion"],
        "safety_triggered": result["safety_triggered"],
        "severity": result.get("severity"),
        "created_at": job_data.get("created_at"),
        "completed_at": datetime.now().isoformat()
    }
    
    redis_client.setex(f"chat_result:{job_id}", 3600, json.dumps(respone_data))        
    
def process_jobs():
    if redis_client is None:
        print("Redis client not available. Cannot process jobs.")
        return
    while True:
        job = redis_client.brpop("chat_jobs")
        if job:
            job_data = json.loads(job[1])
            initiate_chats(job_data)
            
if __name__ == "__main__":
    print("🚀 Worker started, waiting for jobs...")
    process_jobs()