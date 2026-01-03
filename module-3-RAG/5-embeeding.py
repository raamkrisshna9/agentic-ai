from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

pdf_loader = PyPDFLoader("./sample.pdf")
documents = pdf_loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=10,
    length_function=len)
chunks = splitter.split_documents(documents)

# Initialize OpenAI Embeddings model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Generate embeddings for each chunk
# embed_documents method takes a list of strings (chunk contents) and returns a list of embeddings e.g: [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
embeddings = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
print(embeddings)                                               # Print the list of embeddings for each chunk
print(type(embeddings))                                         # <class 'list'> i.e embeddings is a list of embedding vectors for each chunk and each embedding is typically a list of floats. so e.g: [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
print(f"Number of embeddings created: {len(embeddings)}")       # 942 Print number of embeddings created
print(embeddings[0])                                            # Print the first embedding vector for the first chunk
print(type(embeddings[0]))                                      # <class 'list'> i.e each embedding is a list of floats
print(f"Length of 1st embedding vector: {len(embeddings[0])}")  # e.g: 1536 Print length of the first embedding vector


# Iterate through each embedding and its corresponding chunk
for emb,n in zip(embeddings,range(len(embeddings))):
    print(f"Embedding {n+1} --> {emb[:10]} --> Len is {len(emb)}")  # Print first 10 values of each embedding vector of a chunk and its length
