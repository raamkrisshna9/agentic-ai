
#################################################################
# retrieval.py
# Retrieve documents from the Redis vector store based on a query
#################################################################

""" Module for retrieving documents from the Redis vector store based on a query. """

from vector_store import initialize_redis_vector_store, retrieve_similar_documents

# Retrieve documents relevant to a query
def retrieve_documents_for_query(query: str, k: int = 5) -> str:
    context = retrieve_similar_documents(vector_store=initialize_redis_vector_store(), query=query, k=k)
    # Combine the retrieved context (list of documents) into a single string with \n\n separator
    return "\n\n".join(context)

# Note: This function can be used to fetch relevant documents (context, as a single string) from the vector store based on a query,
# which can then be used for further processing, such as providing context to a language model.