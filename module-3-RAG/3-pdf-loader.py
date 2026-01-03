from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                    temperature=0.5, 
                                    max_output_tokens=2048)

# E.g-1: Simple usage of PyPDFLoader to load entire PDF document                         
# Initialise PDF Loader
pdf_loader = PyPDFLoader("./sample.pdf")
documents = pdf_loader.load()
print(documents)
print(len(documents))  # 19,  Print number of Document objects loaded from the PDF, typically one per page
print(documents[0].metadata)  # Print metadata of the first Document object
print(documents[0].page_content)  # Print page content of the first Document object


# lazy_load() function loads pages one by one
pages = []
for doc in pdf_loader.lazy_load():
    pages.append(doc)
    if len(pages) >= 10:
        # do some paged operation, e.g.
        # index.upsert(page)

        pages = []
print(len(pages)) # 9, Print number of Document objects in the last batch

# E.g-2: Using GenericLoader to load a specific PDF file, demonstrating flexibility with different loaders and parsers
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import PyPDFParser

# Initialize GenericLoader with FileSystemBlobLoader and PyPDFParser, to load PDF from filesystem
# FileSystemBlobLoader loads the file as a blob (binary large object + metadata), and PyPDFParser parses the blob into Document objects
generic_loader = GenericLoader(
    blob_loader=FileSystemBlobLoader(
        path="./",
        glob="sample.pdf",
    ),
    blob_parser=PyPDFParser(),
)
documents = generic_loader.load()
print(documents[0].page_content)
print(documents[0].metadata)

# E.g-3: Loading PDF with images using PyPDFLoader with LLMImageBlobParser to handle images in the PDF along with text
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_openai import ChatOpenAI

# Initialize PyPDFLoader with LLMImageBlobParser to parse images in the PDF using an LLM
image_pdf_loader = PyPDFLoader(
    "./sample.pdf",
    mode="page",                         # Load PDF page by page and each page as a Document object
    images_inner_format="markdown-img",
    images_parser=LLMImageBlobParser(model=ChatOpenAI(model="gpt-4o", max_tokens=1024)),
)
documents = image_pdf_loader.load()
print(documents[5].page_content)