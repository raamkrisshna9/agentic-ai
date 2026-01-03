from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader

load_dotenv()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                    temperature=0.5, 
                                    max_output_tokens=2048)

# Initialise Text Loader
text_loader = TextLoader(file_path="./sample.txt")
# load() method returns a list of Document objects for the given text file
# Each document object contains page_content and metadata attributes e.g: Document(page_content=..., metadata={...})
documents = text_loader.load() 
print(type(documents))          # <class 'list'> i.e documents returns a list of Document objects
print(documents)                # Print all the list of documents, typically all the text data will load into a single Document object in the list
print(documents[0])             # Print the first Document object in the list of Document objects


# Output: List of Document objects
# [Document(metadata={'source': './sample.txt'}, 
#           page_content='Agentic AI refers to AI systems that can take autonomous actions, not just generate text.\n
#                         They can plan, reason, make decisions, interact with tools, and take steps toward achieving goals—much like a digital agent')]

print(f"Number of documents loaded: {len(documents)}")  # Print number of Documents loaded

# Iterate through each Document object in the list of Document objects
for doc in documents:
    print(doc.metadata)         # Print metadata of each Document
    print(doc.page_content)     # Print page content of each Document
    print(type(doc))            # <class 'langchain.schema.document.Document'> i.e each Document is of type Document

