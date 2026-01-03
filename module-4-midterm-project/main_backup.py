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

def run_rag_pipeline(query: str) -> str:
    logging.info(f"Running the RAG pipeline for query: {query}")

    # Step-1: Check if the response is in the cache
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Checking if the response is in the cache")
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    cached_response = get_cache(query)
    if cached_response:
        logging.info(f"Cache HIT for query: {query}")
        print(f"\nResponse from cache:\n{cached_response}")
        return cached_response
    logging.info(f"Cache MISS for query: {query}")

    # Step-1b: Semantic cache lookup
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Checking if the response is in the semantic cache")
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    semantic_cached_response, distance = semantic_cache_get(query)
    if semantic_cached_response:
        logging.info(f"Semantic cache HIT for query: {query} (distance={distance})")
        print(f"\nResponse from semantic cache (distance={distance}):\n{semantic_cached_response}")
        set_cache(query, semantic_cached_response)
        return semantic_cached_response
    logging.info(f"Semantic cache MISS for query: {query} (distance={distance})")
    
    # Step-2: Retrieve the context
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Retrieving the context")
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    start_retrieval_time = time.time()
    context = retrieve_similar_documents(vector_store=initialize_redis_vector_store(), query=query)
    end_retrieval_time = time.time()
    retrieval_time = end_retrieval_time - start_retrieval_time
    # retriveal latency is the metric used to measure the time taken to retriving context
    retrieval_latency = int(retrieval_time * 1000)
    logging.info(f"Retrieval latency: {retrieval_latency} ms")
    logging.info(f"Context: {context}")

    # Step-3: Verify context
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    logging.info(f"Verifying the context")
    logging.info(f"++++++++++++++++++++++++++++++++++++++++")
    try:
        is_context_relevant = verify_context(context, query)
        logging.info(f"Context verification: {is_context_relevant}")
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
    return response

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="RAG pipeline")
    parser.add_argument("--query", type=str, required=True, help="Your question to the RAG pipeline")
    args = parser.parse_args()

    # Run the RAG pipeline
    run_rag_pipeline(args.query)
    logging.info("Pipeline completed successfully")







