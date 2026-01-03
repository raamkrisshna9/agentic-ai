#####################################################################
# vector_store.py
# Vector Store Module for RAG (Retrieval-Augmented Generation) System
#####################################################################
""" This module provides functionality for storing and retrieving documents using Redis
as a vector database. It uses OpenAI embeddings to convert text into vector
representations, enabling semantic search capabilities.

Key Components:
- RedisVectorStore: Persistent vector storage using Redis
- OpenAIEmbeddings: Converts text to high-dimensional vectors
- Document: LangChain document structure for storing text content """

import config
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
REDIS_URL = config.REDIS_URL
embedding_model = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

# Initialize Redis Vector Store
def initialize_redis_vector_store():
    return RedisVectorStore(
        redis_url=REDIS_URL,
        index_name=config.INDEX_NAME,
        embeddings=embedding_model,
    )

# Add documents to the vector store
# Accepts a list of strings as documents
def add_documents_to_vector_store(vector_store, documents: list[str]):
    if not vector_store:
        print("Vector store is not initialized.")
        return
    docs = [Document(page_content=doc) for doc in documents]
    try:
        vector_store.add_documents(docs)
        print("Documents added successfully.")
    except Exception as e:
        print(f"Error adding documents: {e}")

# Retrieve similar documents from the vector store, for a given query
# Returns a list of documents contents as a list of strings
def retrieve_similar_documents(vector_store, query: str, k: int = 5):
    if not vector_store:
        print("Vector store is not initialized.")
        return []
    results = vector_store.similarity_search(
        query=query,
        k=k
    )
    return [doc.page_content for doc in results]

# Retrieve similar documents with scores from the vector store, for a given query
# Returns a list of tuples (document content, score)
def retrieve_similar_documents_with_score(vector_store, query: str, k: int = 5):
    if not vector_store:
        print("Vector store is not initialized.")
        return []
    results = vector_store.similarity_search_with_score(
        query=query,
        k=k
    )
    return [(doc.page_content, score) for doc, score in results] # Return list of tuples (document content, score)


