from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from dotenv import load_dotenv

load_dotenv()

pdf_loader = PyPDFLoader("./sample.pdf")
documents = pdf_loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=10,
    length_function=len)
chunks = splitter.split_documents(documents)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# When we use  vectorstores we no need to generate embeddings separately as vectorstores handle that internally.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
# Name of the collection (table) to store vectors
collection_name = "pdf_chunks_collection"

# Initialize PGVector vector store
vector_store = PGVector(
    embeddings=embedding_model,        # This should be embeddings for initializing the vector store
    collection_name=collection_name,  
    connection=connection,
    use_jsonb=True,                   # Store metadata as JSONB
)

# Embed and store the chunks in the vector store
vector_store_embeddings = vector_store.from_documents(
    documents=chunks,
    embedding=embedding_model,        # This should be embedding not embeddings
    collection_name=collection_name,  
    connection=connection,
    use_jsonb=True,                   # Store metadata as JSONB
)



query = "What is Go language?"

# Perform similarity search to retrieve top 5 most similar chunks to the query
# Takes a string query and returns requested number top k most similar Document objects (Chunks)
results = vector_store_embeddings.similarity_search(query, k=5)
print(type(results))  # <class 'list'> i.e results is a list of Document objects e.g: [Document(page_content=..., metadata=...), Document(page_content=..., metadata=...), ...]
print(results)        # Print the list of Document objects retrieved as results
print(f"Number of results retrieved: {len(results)}")  # 5, Print number of results retrieved

# Iterate through each result Document object in the list of results
for i, res in enumerate(results):
    print(f"Result {i+1}:")
    print(res.page_content)
    print("-" * 50)

# Perform similarity search with scores to retrieve top 5 most similar chunks to the query along with their similarity scores
results = vector_store_embeddings.similarity_search_with_score(query, k=5)
print(type(results))  # <class 'list'> i.e results is a list of tuples e.g: [(Document(page_content=..., metadata=...), score), (Document(page_content=..., metadata=...), score), ...]
print(results)        # Print the list of tuples retrieved as results
print(f"Number of results retrieved: {len(results)}")  # 5, Print number of results retrieved

# Iterate through each tuple in the list of results
for i, (res, score) in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Score: {score}")
    print(res.page_content)
    print("-" * 50)

# Note: As the score represents similarity, lower scores indicate higher similarity.