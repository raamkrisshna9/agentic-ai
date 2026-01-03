from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

pdf_loader = PyPDFLoader("./sample.pdf")
documents = pdf_loader.load()
print(type(documents))                                  # <class 'list'> i.e documents is a list of Document objects
print(f"Number of documents loaded: {len(documents)}")  # 19, Print number of Document objects loaded from the PDF

# Initialise RecursiveCharacterTextSplitter, which splits text into chunks based on character count with overlap between chunks to maintain context between them.
# chunk_size: Maximum size of each chunk
# chunk_overlap: Number of overlapping characters between consecutive chunks
# length_function: Function to calculate length of text, here we use len() to count number of characters
splitter = RecursiveCharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=10,
    length_function=len)

# split_documents method splits each Document object from list into list of Document objects, ie. take list of Document objects (pages) as input and returns list of Document objects (chunks) as output.
chunks = splitter.split_documents(documents)
print(chunks)                                       # Print the list of Document objects created after chunking
print(type(chunks))                                 # <class 'list'> i.e chunks is a list of Document objects each representing a chunk
print(f"Number of chunks created: {len(chunks)}")   # 942, Print number of chunks created
print(type(chunks[0]))                              # <class 'langchain.schema.document.Document'> i.e each chunk is of type Document
print(chunks[0])                                    # Print the first chunk Document object

# Iterate through each chunk Document object in the list of chunks
for c,cn in zip(chunks,range(len(chunks))):
   print("chunk details:")
   print(f"Chunk {cn+1} --> {c.page_content} --> chunk length is {len(c.page_content)}")