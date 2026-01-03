#########
# main.py
#########

import time
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    # Write the logs to a file
                    handlers=[logging.FileHandler("rag_pipeline.log", mode="at"),
                    # Write the logs to console
                    logging.StreamHandler()])

from cache_store import get_cache, set_cache
from vector_store import retrieve_similar_documents, initialize_redis_vector_store
from router import build_prompt, verify_context
from llm_client import invoke_model
from semantic_cache import semantic_cache_get, semantic_cache_set
from proximity_retrieval_cache import proximity_cache_get_context, proximity_cache_set_context
import config
from post_process import secured_output
from guard_rails import apply_guardrails
from observability import log, record_metric, start_metrics_server


def run_rag_pipeline(query: str, user_id:str | None = None) -> str:
    logging.info(f"Running the RAG pipeline for query: {query}")

    # Step-1a: Check if the response is in the cache (exact match)
    logging.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Checking if the response is in the exact match cache")
    logging.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    cached_response = get_cache(query)
    if cached_response:
        logging.info(f"Cache HIT for query: {query}")
        print(f"\nResponse from cache:\n{cached_response}")
        return cached_response
    logging.info(f"Cache MISS for query: {query}")

    # Step-1b: Semantic cache lookup
    logging.info(f"+++++++++++++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Checking if the response is in the semantic cache")
    logging.info(f"+++++++++++++++++++++++++++++++++++++++++++++++++++")
    semantic_cached_response, distance = semantic_cache_get(query)
    if semantic_cached_response:
        logging.info(f"Semantic cache HIT for query: {query} (distance={distance})")
        print(f"\nResponse from semantic cache (distance={distance}):\n{semantic_cached_response}")
        set_cache(query, semantic_cached_response)
        return semantic_cached_response
    logging.info(f"Semantic cache MISS for query: {query} (distance={distance})")

    # Step-2a: Proximity (approximate) retrieval cache lookup & retrieval from vector store
    logging.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Checking if the context is in the proximity retrieval cache")
    logging.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    context, context_distance = proximity_cache_get_context(query)
    if context:
        print("Proximity cache HIT for query: {query}")
        logging.info(f"Proximity cache HIT for query: {query} (distance={context_distance})")
        retrieval_latency = 0
        logging.info(f"Retrieval latency: {retrieval_latency} ms")
        logging.info(f"Context: {context}")
    # Step-2b: Retrieve the context from vector store
    else:
        logging.info(f"Proximity cache MISS for query: {query} (distance={context_distance})")
        logging.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        logging.info(f"Retrieving the context from vector store")
        logging.info(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        start_retrieval_time = time.time()

        prompt_k = int(getattr(config, "PROXIMITY_RETRIEVAL_TOP_K", 5))
        rho = int(getattr(config, "PROXIMITY_RERANK_FACTOR", 1))
        overfetch_k = max(prompt_k, prompt_k * max(rho, 1))

        retrieved_docs = retrieve_similar_documents(
            vector_store=initialize_redis_vector_store(),
            query=query,
            k=overfetch_k,
        )

        context = retrieved_docs[:prompt_k]
        proximity_cache_set_context(query, retrieved_docs)

        end_retrieval_time = time.time()
        retrieval_time = end_retrieval_time - start_retrieval_time
        # retriveal latency is the metric used to measure the time taken to retriving context
        retrieval_latency = int(retrieval_time * 1000)
        logging.info(f"Retrieval latency: {retrieval_latency} ms")
        logging.info(f"Context: {context}")
        record_metric("genai_context_retrieval_latency_ms", retrieval_latency)

    # Step-3: Verify context
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Verifying the context")
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    try:
        is_context_relevant = verify_context(context, query)
        logging.info(f"Context verification: {is_context_relevant}")
        # Exist the pipeline if the context is not relevant
        if not is_context_relevant:
            error_msg = "Out of context. Please verify your query and try again."
            print(f"\n{error_msg}")
            logging.info(f"Context verification failed: {error_msg}")
            return error_msg
        logging.info(f"Context verification completed")
    except Exception as e:
        error_msg = "Error verifying context. Please try again."
        logging.error(f"Context verification error: {str(e)}")
        print(f"\n{error_msg}")
        return error_msg

    # Step-4: Build prompt
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Build prompt for the response generation")
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    model_name, prompt = build_prompt(query, context)
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Prompt is ready...")

    # Step-5: Generate the response
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Generating the response....")
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    start_model_time = time.time()
    response = invoke_model(model_name, prompt)
    end_model_time = time.time()
    model_time = end_model_time - start_model_time
    # model latency is the metric used to measure the time taken to get the response from the model
    model_latency = int(model_time * 1000)
    logging.info(f"Model latency: {model_latency} ms")
    logging.info("Response generated..!!!")
    record_metric("genai_model_latency_ms", model_latency)

    # Step-6: Cache the conversation
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Caching the response for query: {query}")
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    set_cache(query, response)
    semantic_cache_set(query, response)
    logging.info(f"Cached response")

    # Step-7: Return the response
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Returning the response")
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Response: {response}")
    print(f"\nGenerated response:\n{response}")
    response = secured_output(response)
    logging.info(f"Secured response: {response}")
    response = apply_guardrails(response)
    logging.info(f"Guardrails applied response: {response}")
    log(query, prompt, response, user_id)
    return response


## Uncomment the below code to run the pipeline as a standalone script
## pipeline.py --query "Your question to the RAG pipeline"

#if __name__ == "__main__":
#    # Start the metrics server
#    start_metrics_server()
#
#    # Parse arguments
#    parser = argparse.ArgumentParser(description="RAG pipeline")
#    parser.add_argument("--query", type=str, required=True, help="Your question to the RAG pipeline")
#    args = parser.parse_args()
#
#    # Run the RAG pipeline
#    run_rag_pipeline(args.query)
#    logging.info("Pipeline completed successfully")
#
#    # Keep the metrics server running
#    try:
#        while True:
#            time.sleep(1)
#    except KeyboardInterrupt:
#        print("Shutting down metrics server...")







