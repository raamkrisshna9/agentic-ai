#################
# api_server.py
#################
# This is the API server for the RAG pipeline

import uvicorn
import time
import uuid 
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from observability import start_metrics_server
from pipeline import run_rag_pipeline

# Initialize the FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    version="1.0.0",
    description="API for the RAG pipeline",
)

# Add CORS middleware, to allow cross origin requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define the reuest model
class AskRequest(BaseModel):
    query: str
    user_id: str | None = None

# Define the response model
class AskResponse(BaseModel):
    response : str
    request_id: str | None = None

# Define the ask endpoint
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    request_id = str(uuid.uuid4())
    response = run_rag_pipeline(request.query, request.user_id)
    return AskResponse(response=response, request_id=request_id)

# Define the metrics endpoint 
@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Define the welcome message endpoint
@app.get("/")
def welcome_message():
    return { "status": "ok", "message": "Welcome to the RAG pipeline API!",
    "metrics_url": f"http://localhost:8005/metrics", "health": f"http://localhost:8005/health"}

# Define the health check endpoint
@app.get("/health")
def health_check():
    return { "status": "ok", "message": "RAG pipeline API is running!"}

# Run the app along with metrics server
if __name__ == "__main__":
    start_metrics_server()
    uvicorn.run(app, host="0.0.0.0", port=8001)