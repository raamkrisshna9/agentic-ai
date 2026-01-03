###########
# router.py
###########
""" This is the place to add prompt engineering related code for routing purposes.
     It can include guardrails (pre-processing) before sending the prompt to the LLM. """

import logging
from config import DEFAULT_MODEL
from llm_client import invoke_model

def verify_context(context: list[str], query: str) -> bool:
    context_verification_template = """You are a context verification assistant. 
    Verify if the provided context is relevant to answer the query.
    
    Context: {context}
    Query: {query}
    
    Respond with exactly 'True' if the context is relevant to the query, or 'False' if not.
    Do not include any other text in your response."""

    context_verification_prompt = context_verification_template.format(context=context, query=query)
    logging.info(f"Context verification started ...")
    response = invoke_model(model_name=DEFAULT_MODEL, prompt=context_verification_prompt)
    
    # Clean and parse the response
    response = response.strip().lower()
    return response == 'true'


# Build the prompt for given query and context, which is extracted using function vector_store.retrieve_similar_documents
def build_prompt(query: str, context: list[str]) -> tuple[str, str]:
    TEMPLATE = """ You are a helpful assistant, answer the following query as best as you can using the provided context.
{context_block}
Query: {query} """

    # if context is not empty, add it to the context block
    context_str = "\n".join(context)
    context_block = f"Context: {context_str}" if context_str.strip() else ""
    model_name = DEFAULT_MODEL
    prompt = TEMPLATE.format(context_block=context_block, query=query)
    return model_name, prompt