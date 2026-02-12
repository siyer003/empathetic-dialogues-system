# Empathetic Dialogue Chatbot

A Dockerized empathetic chatbot built with FastAPI, Redis, and transformer-based models.  
Implements asynchronous inference with server-managed session history.

---

## Features

- Transformer-based empathetic response generation  
- Emotion-aware dialogue modeling  
- Asynchronous job processing via Redis  
- Session-based conversation history (TTL-enabled)  
- Fully containerized multi-service architecture  

---

## Architecture

```
Client → FastAPI → Redis Queue → Worker → Redis (Session Store)
```

- **API Service**: Accepts chat requests and enqueues jobs  
- **Worker Service**: Performs emotion detection and response generation  
- **Redis**: Handles job queue and session memory  

---

## Run with Docker

### Build and start services

```bash
docker compose up --build
```

### Open Swagger UI

```
http://localhost:8000/docs
```

---

## Chat Flow

### Start a chat

**POST /chat**

```json
{
  "message": "I'm having a bad day",
  "user_id": "anonymous",
  "session_id": null
}
```

**Response**

```json
{
  "job_id": "...",
  "session_id": "...",
  "status": "queued"
}
```

### Get the response

**GET /response/{job_id}**

Reuse the same `session_id` to continue the conversation.

---

## Session Management

Session history is stored in Redis using:

```
session:{session_id}:history
```

- Only the last 10 turns are retained
- Sessions expire after 1 hour of inactivity

---

## Project Structure

```
.
├── Dockerfile.api
├── Dockerfile.worker
├── docker-compose.yml
├── services/
│   ├── api.py
│   ├── worker.py
│   └── utils.py
├── requirements-api.txt
├── requirements-worker.txt
```

---

## Tech Stack

- Python 3.12
- FastAPI
- Redis
- Hugging Face Transformers
- Docker & Docker Compose

---

## Future Improvements

- Deploy to AWS ECS / EKS
- Add monitoring and metrics
- Add frontend UI
- Persist sessions in PostgreSQL
