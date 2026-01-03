
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager
from langchain_core.indexing import index
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

## Step-1: Data loading and chunking
#===================================
# Load PDF with images using PyPDFLoader with LLMImageBlobParser to handle images in the PDF along with text
# LLMImageBlobParser uses an LLM to parse images in the PDF
image_pdf_loader = PyPDFLoader(
    "./Ebook-Agentic-AI.pdf",
    mode="page",                         # Load PDF page by page and each page as a Document object
    images_inner_format="markdown-img",  # Format to represent images in the Document content
    images_parser=LLMImageBlobParser(model=ChatOpenAI(model="gpt-4o", max_tokens=1024)),
)
documents = image_pdf_loader.load()

# Initialize RecursiveCharacterTextSplitter and split the documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
    length_function=len)
chunks = splitter.split_documents(documents)

## Step-2: Indexing along with Record Manager and Vector Store
#=============================================================
# Initialize OpenAI Embeddings model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize Chroma vector store loaded with the chunks
collection_name = "my_collection_1"
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory="./chroma_langchain_1_db",
)
vector_store.from_documents(chunks)

# Initialize SQLRecordManager and create schema
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"
namespace = "my_namespace"
record_manager = SQLRecordManager(
    db_url=connection,
    namespace=namespace,
)
record_manager.create_schema()

# Create the index using the chunks, vector store, and record manager.
# index function creates an index that integrates the vector store and record manager for efficient retrieval and storage.
index = index(chunks, vector_store=vector_store, record_manager=record_manager, cleanup="incremental", source_id_key="source")

## Step-3: Retrieval
#===================
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})  # Retrieve top 5 most similar chunks
query = "who is PRASENJIT DEY?"
retrieved_docs = retriever.invoke(query)

## Step-4: Response Generation
#=============================
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.5)
prompt = ChatPromptTemplate.from_template(
    """"Answer the question based only on the provided context.
    context: {context}
    question: {question}"""
)
llm_chain = prompt | llm

user_input = {"context": retrieved_docs,
              "question":query }
Response = llm_chain.invoke(user_input)

print("Output from RAG:")
print("------------------ \n")
print(Response.content)


