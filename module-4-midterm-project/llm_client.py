##############
#llm_client.py
##############

""" This is the LLM client for the RAG pipeline, it uses langchain to invoke the model """

from langchain_openai import ChatOpenAI
from config import TEMPERATURE, MAX_TOKENS
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
import logging
from functools import lru_cache

load_dotenv()

# Initialize the chat model, pass the model name, temperature and max tokens
# lru_cache is used to cache the results of the function to avoid reinitializing the chat model for the same model name
@lru_cache(maxsize=10)
def _initilise_model(model_name: str) -> ChatOpenAI:
    logging.info(f"Initializing the chat model for {model_name}")
    return ChatOpenAI(model=model_name, 
            temperature=TEMPERATURE, 
            max_tokens=MAX_TOKENS
            )

# Invoke the model, pass the model name and prompt
def invoke_model(model_name:str, prompt: str) -> str:
    # Initialize the model
    model = _initilise_model(model_name)
    logging.info(f"Calling the model {model_name}")
    response = model.invoke(prompt)
    # Type cast the response to AIMessage (e.g: x:int = 10)
    response:AIMessage = response
    return response.content