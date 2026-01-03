###################
# fast_api_demo.py
###################

""" This is a simple demo of FastAPI, it has 4 endpoints
 1. / - root endpoint/homepage
 2. /metrics - metrics endpoint
 3. /rag - rag endpoint
 4. /generate - generate endpoint
 FastAPI : Its a web framework for building APIs, converts the python code to REST APIs
 Benefits of using FastAPI:
 - Automatically create API documentation using Swagger UI and expose at /docs endpoint
 - Can use pydantic models to validate the input and output
 - Great for model serving and API development
 - It is fast to code as compared to other frameworks like Flask, Django, etc.
 - It is easy to use and has a lot of features like authentication, authorization, etc.
 - It is easy to scale and deploy. """

from fastapi import FastAPI
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Root endpoint, when invoke http://localhost:8005/, runs the welcome_message function and returns a welcome message
@app.get("/")
def welcome_message():
    return {"message": "Welcome to the RAG pipeline!"}

# Metrics endpoint, when invoke http://localhost:8005/metrics, runs the metrics function
@app.get("/metrics")    
def metrics():
    return {"message": "Metrics are available here"}

# RAG endpoint, when invoke http://localhost:8005/rag, runs the rag_pipeline function
@app.get("/rag")
def rag_pipeline():
    return {"message": "RAG pipeline is running!"}

# Generate endpoint, when invoke http://localhost:8005/generate, runs the generate_response function
@app.post("/generate")
def generate_response(question: str):
    output = question[::-1]
    return {"output": f"the generated response is {output}"}

# Run the app, when run the app it will start the server at http://localhost:8005
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)