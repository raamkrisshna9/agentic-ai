###################
# observability.py
# Promethus having metrics types Counter, Histogram, Summary, Gauge 
###################

""" This is the observability module for the RAG pipeline, it uses prometheus to expose metrics """

import logging
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
REQUEST_COUNTER = Counter("genai_requests_total", "Total requests received")
MODEL_LATENCY = Histogram("genai_model_latency_ms", "Model call latency in milliseconds")
CONTEXT_RETRIEVAL_LATENCY = Histogram("genai_context_retrieval_latency_ms", "Context retrieval step latency in milliseconds")

#  Use this as a central logging function for the entire pipeline
def log(query, model_input, model_output, user_id=None):
    REQUEST_COUNTER.inc()
    logging.info("Number of requests: {REQUEST_COUNTER}")
    logging.info(f"Query: {query}")
    logging.info(f"Model input: {model_input}")
    logging.info(f"Model output: {model_output}")
    logging.info(f"User ID: {user_id if user_id else 'N/A'}")
    logging.info("---------------------------------")

# This function records the metrics
def record_metric(metric_name, value):
    if metric_name == "genai_model_latency_ms":
        MODEL_LATENCY.observe(value)
    elif metric_name == "genai_context_retrieval_latency_ms":
        CONTEXT_RETRIEVAL_LATENCY.observe(value)

# This function starts the metrics server, expose the metrics at http://localhost:8002/metrics endpoint
def start_metrics_server(port=8002):
    start_http_server(port)
    logging.info(f"Prometheus metrics server running at http://localhost:{port}/metrics")