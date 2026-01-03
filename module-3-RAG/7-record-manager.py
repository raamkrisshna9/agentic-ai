

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager
from langchain_core.indexing import index
from dotenv import load_dotenv

load_dotenv()

text_loader = TextLoader("./sample_1.txt")
documents = text_loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=10,
    length_function=len)
chunks = splitter.split_documents(documents)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
collection_name = "my_collection"
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory="./chroma_langchain_db",
)
vector_store.from_documents(chunks)

# Initialize SQLRecordManager
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"
namespace = "my_namespace"
record_manager = SQLRecordManager(
    db_url=connection,
    namespace=namespace,
)
# Create the schema in the database, which sets up necessary tables for storing records
record_manager.create_schema()
print(record_manager.list_keys())


# Create the index using the chunks, vector store, and record manager
index = index(chunks, vector_store=vector_store, record_manager=record_manager, cleanup="incremental", source_id_key="source")
print(f"Index created with {len(chunks)} documents.", index)

#output: Index created with 17 documents. {'num_added': 3, 'num_updated': 0, 'num_skipped': 14, 'num_deleted': 0}



