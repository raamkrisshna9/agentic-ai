################################################################################
# test_retrieval.py
# Test retrieval of documents with similarity scores from the Redis vector store
################################################################################
"""Test retrieval of documents with similarity scores from the Redis vector store."""

from vector_store import initialize_redis_vector_store, retrieve_similar_documents_with_score

query = "What is Autonomy in Agentic AI?"
# Retrieve similar documents with scores
results = retrieve_similar_documents_with_score(vector_store=initialize_redis_vector_store(), query=query, k=3)
for doc, score in results:
    print(f"Score: {score:.4f}\nDocument: {doc}\n")

# Notes: Sometimes, we may endup with similar documents with very close similarity scores, whcih requires further processing (Response filtering)
# Response Filtering - To avoid duplicate results
### if similarity difference is < 0.001, then consider it as the same document
### MMR - Maximal Marginal Relevance
### Reduce chunk overlap